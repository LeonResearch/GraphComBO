#!/bin/bash

ARGS_KERNELS=(
    "--label diffusion_ard"
    "--label baselines"
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
ARGS_PROBLEM=(
    "--problem BA"
    "--problem WS"
    "--problem GRID"
)
ARGS_STARTING=(
    "--starting random"
)
ARGS_STARTING=(
    "--start_seed 0"
    "--start_seed 5"
    "--start_seed 10"
    "--start_seed 15"
    )
ARGS_EXPLOITATION=(
    "--exploitation True"
)
ARGS_RESTART=(
    "--restart same_as_start"
    "--restart queried_best"
    )
starting_core=0
n_cores_per_job=1

echo ${#ARGS_KERNELS[@]}
# Loop through the Python files and run them in parallel with taskset and specified CPU cores
for ((i=0; i<${#ARGS_K[@]}; i++)); do
    for ((j=0; j<${#ARGS_PROBLEM[@]}; j++)); do
        for ((k=0; k<${#ARGS_KERNELS[@]}; k++)); do
            # Calculate the idx of CPU cores for each job
            START_CORE=$(( (i*${#ARGS_PROBLEM[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + k*n_cores_per_job + starting_core ))
            END_CORE=$(( (i*${#ARGS_PROBLEM[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + (k+1)*n_cores_per_job + starting_core-1 ))
            CPU_CORES=${START_CORE}-${END_CORE}
            echo "CPU Cores ${CPU_CORES} for ${ARGS_K[i]} ${ARGS_PROBLEM[j]} ${ARGS_KERNELS[k]}"
            # Use taskset to run the python processes with specified cores
            taskset -c ${CPU_CORES} python main.py ${ARGS_K[i]} ${ARGS_PROBLEM[j]} ${ARGS_KERNELS[k]} ${ARGS_STARTING[0]} ${ARGS_EXPLOITATION[0]} &
        done
    done
done

starting_core=24
# Loop through the Python files and run them in parallel with taskset and specified CPU cores
for ((i=0; i<${#ARGS_K[@]}; i++)); do
    for ((j=0; j<${#ARGS_PROBLEM[@]}; j++)); do
        for ((k=0; k<${#ARGS_KERNELS[@]}; k++)); do
            # Calculate the idx of CPU cores for each job
            START_CORE=$(( (i*${#ARGS_PROBLEM[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + k*n_cores_per_job + starting_core ))
            END_CORE=$(( (i*${#ARGS_PROBLEM[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + (k+1)*n_cores_per_job + starting_core-1 ))
            CPU_CORES=${START_CORE}-${END_CORE}
            echo "CPU Cores ${CPU_CORES} for ${ARGS_K[i]} ${ARGS_PROBLEM[j]} ${ARGS_KERNELS[k]}"
            # Use taskset to run the python processes with specified cores
            taskset -c ${CPU_CORES} python main.py ${ARGS_K[i]} ${ARGS_PROBLEM[j]} ${ARGS_KERNELS[k]} ${ARGS_STARTING[1]} ${ARGS_EXPLOITATION[0]} &
        done
    done
done

starting_core=48
# Loop through the Python files and run them in parallel with taskset and specified CPU cores
for ((i=0; i<${#ARGS_K[@]}; i++)); do
    for ((j=0; j<${#ARGS_PROBLEM[@]}; j++)); do
        for ((k=0; k<${#ARGS_KERNELS[@]}; k++)); do
            # Calculate the idx of CPU cores for each job
            START_CORE=$(( (i*${#ARGS_PROBLEM[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + k*n_cores_per_job + starting_core ))
            END_CORE=$(( (i*${#ARGS_PROBLEM[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + (k+1)*n_cores_per_job + starting_core-1 ))
            CPU_CORES=${START_CORE}-${END_CORE}
            echo "CPU Cores ${CPU_CORES} for ${ARGS_K[i]} ${ARGS_PROBLEM[j]} ${ARGS_KERNELS[k]}"
            # Use taskset to run the python processes with specified cores
            taskset -c ${CPU_CORES} python main.py ${ARGS_K[i]} ${ARGS_PROBLEM[j]} ${ARGS_KERNELS[k]} ${ARGS_STARTING[2]} ${ARGS_EXPLOITATION[0]} &
        done
    done
done

starting_core=72
# Loop through the Python files and run them in parallel with taskset and specified CPU cores
for ((i=0; i<${#ARGS_K[@]}; i++)); do
    for ((j=0; j<${#ARGS_PROBLEM[@]}; j++)); do
        for ((k=0; k<${#ARGS_KERNELS[@]}; k++)); do
            # Calculate the idx of CPU cores for each job
            START_CORE=$(( (i*${#ARGS_PROBLEM[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + k*n_cores_per_job + starting_core ))
            END_CORE=$(( (i*${#ARGS_PROBLEM[@]}*${#ARGS_KERNELS[@]})*n_cores_per_job + (j*${#ARGS_KERNELS[@]})*n_cores_per_job + (k+1)*n_cores_per_job + starting_core-1 ))
            CPU_CORES=${START_CORE}-${END_CORE}
            echo "CPU Cores ${CPU_CORES} for ${ARGS_K[i]} ${ARGS_PROBLEM[j]} ${ARGS_KERNELS[k]}"
            # Use taskset to run the python processes with specified cores
            taskset -c ${CPU_CORES} python main.py ${ARGS_K[i]} ${ARGS_PROBLEM[j]} ${ARGS_KERNELS[k]} ${ARGS_STARTING[3]} ${ARGS_EXPLOITATION[0]} &
        done
    done
done
# Wait for all background processes to finish
wait
