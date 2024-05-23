#!/bin/bash

ARGS_KERNELS=(
    "--label diffusion_ard"
    "--label local_search"
    "--label bfs"
    "--label dfs"
    "--label random"
    #"--label baselines"
    #"--label diffusion"
    #"--label polynomial_suminverse"
    #"--label polynomial"
)
ARGS_K=(
    "--k 4"
    "--k 8"
    "--k 16"
    "--k 32"
)
ARGS_STARTING=(
    "--starting random"
    "--starting degree"
    )
ARGS_PROBLEM=(
    "--problem Road_IC"
)
ARGS_EXPLOITATION=(
    "--exploitation True"
    )
ARGS_RESTART=(
    "--restart same_as_start"
    "--restart queried_best"
    )
starting_core=40
n_cores_per_job=1
echo ${#ARGS_KERNELS[@]}
# Loop through the Python files and run them in parallel with taskset and specified CPU cores
for ((i=0; i<${#ARGS_K[@]}; i++)); do
    for ((j=0; j<${#ARGS_STARTING[@]}; j++)); do
        for ((k=0; k<${#ARGS_KERNELS[@]}; k++)); do
            # Calculate the idx of CPU cores for each job
            START_CORE=$(( (i*${#ARGS_STARTING[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + k*n_cores_per_job + starting_core ))
            END_CORE=$(( (i*${#ARGS_STARTING[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + (k+1)*n_cores_per_job + starting_core-1 ))
            CPU_CORES=${START_CORE}-${END_CORE}
            echo "CPU Cores ${CPU_CORES} for ${ARGS_K[i]} ${ARGS_STARTING[j]} ${ARGS_KERNELS[k]}"
            # Use taskset to run the python processes with specified cores
            taskset -c ${CPU_CORES} python main.py ${ARGS_K[i]} ${ARGS_STARTING[j]} ${ARGS_KERNELS[k]} ${ARGS_PROBLEM[0]} ${ARGS_EXPLOITATION[0]} &
        done
    done
done

# Wait for all background processes to finish
wait

