from turtle import forward
import gpytorch.kernels
from gpytorch.kernels import Kernel
import torch
from typing import Optional
from .utils import eigendecompose_laplacian
import networkx as nx


class DiffusionGraphKernel(Kernel):
    has_lengthscale = True

    def __init__(
        self,
        context_graph: nx.Graph = None,
        eigenbasis: Optional[torch.Tensor] = None,
        eigenvalues: Optional[torch.Tensor] = None,
        precompute_eigendecompose: bool = True,
        normalized_laplacian: bool = True,
        order: int = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # check dimensions
        assert context_graph is not None or (
            eigenbasis is not None and eigenvalues is not None
        )

        if eigenvalues is None or eigenbasis is None:
            if precompute_eigendecompose:
                eigenvalues, eigenbasis = eigendecompose_laplacian(
                    context_graph, normalized_laplacian=normalized_laplacian
                )
            else:
                eigenvalues, eigenbasis = None, None
        else:
            assert eigenvalues.ndimension() == 1
        self.eigenbasis = eigenbasis
        self.eigenvalues = eigenvalues
        self.order = order

    def get_dist(self):
        order = (
            min(self.order, self.eigenvalues.shape[0])
            if self.order
            else self.eigenvalues.shape[0]
        )
        effective_eigenvalues = self.eigenvalues[:order]
        dists = torch.exp(-effective_eigenvalues * self.lengthscale)
        if order > 1:
            dists = torch.diag(dists.squeeze())
            dists *= order / torch.sum(dists)
        # else:
        #     dists = dists.unsqueeze(0)
        return dists

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: torch.Tensor of shape (b1 x ... x bn x n x 2): each element is an edge index.
        x2: torch.Tensor of shape (b1 x ... x bn x m x 2): each element is an edge index.
        Output:
            kernel matrix of dim (b1 x .... x bn x n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError("Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        order = (
            min(self.order, self.eigenvalues.shape[0])
            if self.order
            else self.eigenvalues.shape[0]
        )
        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        # b1 x ...x bn x n x N
        subvec1 = self.eigenbasis[x1_, :order]
        # b1 x ...x bn x m x N
        subvec2 = self.eigenbasis[x2_, :order]
        dists = self.get_dist()  # N x N
        self._dists = torch.diagonal(dists.clone(), 0)

        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)

        return res


class PolynomialKernel(DiffusionGraphKernel):
    has_lengthscale = True

    def get_dist(self):
        epsilon = 1e-6
        order = (
            min(self.order, self.eigenvalues.shape[0])
            if self.order
            else self.eigenvalues.shape[0]
        )
        effective_eigenvalues = self.eigenvalues[:order]
        # Note the definition of the B matrix here -- we directly power the eigenvalues
        # without the inversion in the previous iteration.
        eigen_powered = torch.cat(
            [(effective_eigenvalues**i).reshape(1, -1) for i in range(self.order)]
        )  # shape: (self.order, n)
        # This is the B matrix
        # dists = torch.einsum("ij,i->ij", eigen_powered,self.beta.squeeze(0))
        # Sum B matrix
        dists = torch.einsum("ij,i->ij", eigen_powered, self.lengthscale.squeeze(0))
        dists = torch.diag(1 / (dists.sum(0).squeeze() + epsilon))
        # print(dists, self.beta)
        return dists

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: torch.Tensor of shape (b1 x ... x bn x n x 1): each element is a vertice index.
        x2: torch.Tensor of shape (b1 x ... x bn x m x 1): each element is a vertice index.
        Output:
            kernel matrix of dim (b1 x .... x bn x n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError("Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        order = (
            min(self.order, self.eigenvalues.shape[0])
            if self.order
            else self.eigenvalues.shape[0]
        )
        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        # b1 x ...x bn x n x N
        subvec1 = self.eigenbasis[x1_, :order]
        # b1 x ...x bn x m x N
        subvec2 = self.eigenbasis[x2_, :order]
        dists = self.get_dist()  # N x N
        self._dists = torch.diagonal(dists.clone(), 0)

        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


class PolynomialKernelSumInverse(DiffusionGraphKernel):
    has_lengthscale = True

    def get_dist(self):
        epsilon = 1e-6
        order = (
            min(self.order, self.eigenvalues.shape[0])
            if self.order
            else self.eigenvalues.shape[0]
        )
        effective_eigenvalues = self.eigenvalues[:order]
        eigen_powered = torch.stack([effective_eigenvalues**i for i in range(self.order)])

        dists = 1.0 / (
            torch.einsum("ij,i->ij", eigen_powered, self.lengthscale.squeeze(0))
            + epsilon
        )
        dists = torch.diag(dists.sum(0).squeeze())
        dists *= effective_eigenvalues.shape[0] / torch.sum(dists)
        return dists

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: torch.Tensor of shape (b1 x ... x bn x n x 1): each element is a vertice index.
        x2: torch.Tensor of shape (b1 x ... x bn x m x 1): each element is a vertice index.
        Output:
            kernel matrix of dim (b1 x .... x bn x n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError("Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        order = (
            min(self.order, self.eigenvalues.shape[0])
            if self.order
            else self.eigenvalues.shape[0]
        )
        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        # b1 x ...x bn x n x N
        subvec1 = self.eigenbasis[x1_, :order]
        # b1 x ...x bn x m x N
        subvec2 = self.eigenbasis[x2_, :order]
        dists = self.get_dist()  # N x N
        self._dists = torch.diagonal(dists.clone(), 0)

        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res
