import numpy as np
import networkx as nx
import torch
import gpytorch
import botorch
from torch import Tensor
from typing import Tuple, Union, Optional, Dict, Any
from time import time
from math import log
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import GammaPrior
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP, SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import standardize
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    NoisyExpectedImprovement,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    UpperConfidenceBound,
    qUpperConfidenceBound,
)
from .kernels import DiffusionGraphKernel, PolynomialKernel, PolynomialKernelSumInverse
from .utils import eigendecompose_laplacian, fit_gpytorch_model, filter_invalid


def initialize_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    context_graph: nx.Graph,
    covar_type: str = "polynomial",
    covar_kwargs: Optional[Dict[str, Any]] = None,
    use_fixed_noise: bool = False,
    fit_model: bool = False,
    ard: bool = False,
    cached_eigenbasis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_normalized_laplacian: bool = True,
    use_normalized_eigenvalues: bool = True,
    use_saas_map: bool = False,
    n_cv_fold: int = -1,
    taus: Optional[torch.Tensor] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    standardize_y: bool = True,
):
    if cached_eigenbasis is None:
        laplacian_eigenvals, laplacian_eigenvecs = eigendecompose_laplacian(
            context_graph,
            normalized_laplacian=use_normalized_laplacian,
            normalized_eigenvalues=use_normalized_eigenvalues,
        )
    else:
        laplacian_eigenvals, laplacian_eigenvecs = cached_eigenbasis
    cached_eigenbasis = (laplacian_eigenvals, laplacian_eigenvecs)
    if standardize_y:
        train_Y = standardize(train_Y.clone())
    if use_fixed_noise:
        train_Yvar = torch.full_like(train_Y, 1e-7) * train_Y.std(dim=0).pow(2)
    covar_kwargs = covar_kwargs or {}
    model_kwargs = []
    base_model_class = (
        FixedNoiseGP
        if (use_fixed_noise and not torch.isnan(train_Yvar).any())
        else SingleTaskGP
    )

    if covar_type in ["polynomial", "diffusion", "polynomial_suminverse"]:
        if covar_type == "polynomial":
            base_covar_class = PolynomialKernel
        elif covar_type == "diffusion":
            base_covar_class = DiffusionGraphKernel
        else:
            base_covar_class = PolynomialKernelSumInverse
        order = covar_kwargs.get("order", None)
        # when order is not explicitly specified,
        if covar_type == "diffusion":
            order = min(order, train_X.shape[-2]) if order else len(context_graph)
        """
        elif covar_type in ["polynomial", "polynomial_suminverse"]:
            if order == None:
                order = min(5, nx.radius(context_graph))
        """
        if ard:
            ard_num_dims = order
        else:
            ard_num_dims = None
        covar_kwargs.update({"ard_num_dims": ard_num_dims, "order": order})
    else:
        raise NotImplementedError(f"covar_type {covar_type} is not implemented.")

    if "lengthscale_constraint" not in covar_kwargs.keys():
        covar_kwargs["lengthscale_constraint"] = GreaterThan(1e-5)

    if train_Y.shape[-1] > 1:
        raise NotImplementedError(
            f"Multi - objective search is not currently supported."
            "train_Y has last dimension"
            "of {train_Y.shape[-1]}!"
        )
    model_kwargs.append(
        {
            "train_X": train_X,
            "train_Y": train_Y,
            "covar_module": gpytorch.kernels.ScaleKernel(
                base_covar_class(
                    eigenvalues=laplacian_eigenvals,
                    eigenbasis=laplacian_eigenvecs,
                    **covar_kwargs,
                )
            ),
        }
    )
    if use_fixed_noise and not torch.isnan(train_Yvar).any():
        model_kwargs[0]["train_Yvar"] = train_Yvar
    else:
        model_kwargs[0]["likelihood"] = GaussianLikelihood(
            noise_prior=GammaPrior(0.9, 10.0), noise_constraint=Interval(1e-7, 1e-3)
        )

    # create model
    models = [base_model_class(**model_kwargs[i]) for i in range(len(model_kwargs))]
    if len(models) > 1:
        model = ModelListGP(*models).to(device=train_X.device, dtype=train_X.dtype)
        mll = SumMarginalLogLikelihood(model.likelihood, model).to(
            device=train_X.device
        )
    else:
        model = models[0].to(device=train_X.device, dtype=train_X.dtype)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(
            device=train_X.device
        )
    if fit_model:
        optim_options = dict(lr=0.1, train_iters=100)
        optim_options.update(optim_kwargs or {})
        with gpytorch.settings.debug(False):
            fit_gpytorch_model(mll, model, train_X, train_Y, **optim_options)
    return model, mll, cached_eigenbasis


def get_acqf(
    model,
    X_baseline: Tensor,
    train_Y: Tensor,
    batch_size: int = 1,
    acq_type: str = "ei",
    mc_samples: int = 1024,
    ucb_beta: Optional[float] = 0.1,
    iteration: Optional[int] = None,
    ref_point: Optional[Tensor] = None,
):
    if acq_type == "ucb":
        if ucb_beta:
            beta = ucb_beta
        else:
            assert iteration is not None
            beta = 0.2 * X_baseline.shape[-1] * log(2 * iteration)
    if acq_type in ["ei", "ucb", "logei"]:
        assert train_Y.shape[1] == 1
        if batch_size == 1:
            if acq_type == "ei":
                acq_func = ExpectedImprovement(
                    model,
                    best_f=train_Y.max(),
                )
            if acq_type == "logei":
                acq_func = LogExpectedImprovement(
                    model,
                    best_f=train_Y.max(),
                )
            elif acq_type == "nei":
                acq_func = NoisyExpectedImprovement(model, X_observed=X_baseline)
            elif acq_type == "ucb":
                acq_func = UpperConfidenceBound(model, beta=beta)
        else:
            if acq_type == "ei":
                acq_func = qExpectedImprovement(
                    model=model,
                    best_f=train_Y.max(),
                    sampler=SobolQMCNormalSampler(mc_samples),
                )
            elif acq_type == "nei":
                acq_func = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=X_baseline,
                    sampler=SobolQMCNormalSampler(mc_samples),
                    prune_baseline=True,
                )
            elif acq_type == "ucb":
                acq_func = qUpperConfidenceBound(
                    model=model, beta=beta, sampler=SobolQMCNormalSampler(mc_samples)
                )
    return acq_func


def optimize_acqf(
    model,
    acqf: botorch.acquisition.AcquisitionFunction,
    context_graph: Union[nx.Graph, Tuple[Tensor, Tensor]],
    method: str = "enumerate",
    batch_size: int = 1,
    noisy: bool=False,
    options: Optional[Dict[str, Any]] = None,
    X_avoid: Optional[torch.Tensor] = None,
    X_prior: Optional[torch.Tensor] = None,
    acq_type: str = None,
):
    assert method in ["enumerate", "local_search"]
    nodes_to_eval = []
    if method == "enumerate":
        # enumerate the acquisition function of all nodes in the context graph
        if isinstance(context_graph, nx.Graph):
            nnodes = len(context_graph.nodes)
        else:
            nnodes = context_graph[0].shape[0]
        # get all possible nodes in the current graph
        all_possible_nodes = torch.arange(nnodes).reshape(-1, 1)
        if noisy: # generate posterior mean if under noisy setting
            model.eval()
            posterior = model.posterior(all_possible_nodes)
            mean = posterior.mean
        best_posterior_mean_node = None

        if X_avoid is not None:
            all_possible_nodes = filter_invalid(all_possible_nodes, X_avoid)
            # print('# testing nodes in current context graph:', len(all_possible_nodes))
            if not all_possible_nodes.shape[0]:
                return None, None
        for q in range(batch_size):
            acqf_vals = acqf(all_possible_nodes.unsqueeze(1))
            if X_prior is not None:
                X_prior = X_prior[all_possible_nodes].squeeze()
                acqf_vals = acqf_vals * X_prior
            print(f'Acquisition max value: {acqf_vals.max().item()}')
            best_node = torch.argmax(acqf_vals).item()
            nodes_to_eval.append(all_possible_nodes[best_node])
    elif method == "local_search":
        default_options = {
            "num_restarts": 5,
            "device": "cpu",
        }
        default_options.update(options or {})
        nodes_to_eval, _, _ = local_search(
            acqf,
            context_graph,
            q=batch_size,
            **default_options,
        ).tolist()
    return torch.tensor(nodes_to_eval), best_posterior_mean_node
