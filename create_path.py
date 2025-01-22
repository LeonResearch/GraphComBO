import os

def create_path(save_dir, problem_name, problem_kwargs, bo_kwargs):
    if problem_name == "epidemic":
        if problem_kwargs["underlying_function"] == 'population_infection_time':
            s = "_".join([problem_kwargs["graph_type"],
                          problem_kwargs["underlying_function"], 
                          f'k-{problem_kwargs["k"]}',
                          f'threshold-{problem_kwargs["infection_percentage_threshold"]}',
                          f'samples-{problem_kwargs["SIR_n_samples"]}',
                          f'iter-{problem_kwargs["SIR_n_iterations"]}'
                          ])
        elif problem_kwargs["underlying_function"] == 'individual_infection_time':
            s = "_".join([problem_kwargs["graph_type"],
                          problem_kwargs["underlying_function"], 
                          f'k-{problem_kwargs["k"]}',
                          f'iter-{problem_kwargs["SIR_n_iterations"]}'
                          ])
    elif problem_name == "influence_maximisation":
        s = "_".join([problem_kwargs["graph_type"],
                      problem_kwargs["underlying_function"], 
                      f'k-{problem_kwargs["k"]}',
                      f'ICp-{problem_kwargs["IC_p"]}',
                      f'samples-{problem_kwargs["IC_n_samples"]}'
                      ])
    elif problem_name == "resilience":
        s = "_".join([problem_kwargs["graph_type"],
                      problem_kwargs["underlying_function"], 
                      f'k-{problem_kwargs["k"]}',
                      ])
    elif problem_name == "gnn_attack":
        s = "_".join([problem_kwargs["graph_type"],
                      problem_kwargs["underlying_function"], 
                      f'idx-{problem_kwargs["graph_index"]}',
                      f'k-{problem_kwargs["k"]}',
                      ])
    elif problem_name == "synthetic":
        if problem_kwargs["graph_type"] == "ba":
            s = "_".join([problem_kwargs["graph_type"], 
                          problem_kwargs["underlying_function"], 
                          f'n-{problem_kwargs["n"]}',
                          f'k-{problem_kwargs["k"]}',
                          f'm-{problem_kwargs["m"]}', 
                          ])
        elif problem_kwargs["graph_type"] == "ws":
            s = "_".join([problem_kwargs["graph_type"], 
                          problem_kwargs["underlying_function"], 
                          f'n-{problem_kwargs["n"]}',
                          f'k-{problem_kwargs["k"]}',
                          f'wsk-{problem_kwargs["wsk"]}', f'p-{problem_kwargs["p"]}', 
                          ])
        elif problem_kwargs["graph_type"] == "sbm":
            s = "_".join([problem_kwargs["graph_type"], 
                          problem_kwargs["underlying_function"], 
                          f'n-{problem_kwargs["n"]}',
                          f'k-{problem_kwargs["k"]}',
                          f'ngroup-{problem_kwargs["ngroup"]}', 
                          f'probin-{problem_kwargs["probin"]}', 
                          f'probout-{problem_kwargs["probout"]}', 
                          ])
        elif problem_kwargs["graph_type"] == "grid":
            s = "_".join([problem_kwargs["graph_type"], 
                          problem_kwargs["underlying_function"], 
                          f'n-{problem_kwargs["n"]}',
                          f'k-{problem_kwargs["k"]}',
                          f'noise-{problem_kwargs["noise"]}', 
                          ])
        elif problem_kwargs["graph_type"] == "ogb_arxiv":
            s = "_".join([problem_kwargs["graph_type"], 
                          problem_kwargs["underlying_function"], 
                          f'k-{problem_kwargs["k"]}',
                          ])
        if getattr(problem_kwargs, "noisy", False):
            s = "_".join([s, f'noise-{problem_kwargs["noise"]}'])

    s = "_".join([s, 
                  f'ninit-{bo_kwargs["n_init"]}',
                  f'failtol-{bo_kwargs["tr_settings"]["fail_tol"]}', 
                  f'Q-{bo_kwargs["Q"]}',
                  f'exploitation-{bo_kwargs["exploitation"]}',
                  f'l-{bo_kwargs["max_radius"]}',
                  f'start-{bo_kwargs["start_location"]}',
                  f'restart-{bo_kwargs["restart_location"]}',
                  f'query-{bo_kwargs["max_iters"]}'])
    save_path = os.path.join(save_dir, s)
    return save_path
