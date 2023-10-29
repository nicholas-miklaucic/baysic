#!/bin/bash
#SBATCH --job-name=baysic
#SBATCH --output /dev/null
#SBATCH --error /dev/null
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --time 1-0
#SBATCH -p defq-48core, defq-64-core

source /home/miklaucn/bin/baysic

file="/home/miklaucn/mp20_train_ids.txt"  # Replace with the path to your file

start=${SLURM_ARRAY_TASK_ID:-1}
increment=${SLURM_ARRAY_TASK_COUNT:-10000}

current_line=1

start=`date +%s`
while IFS= read -r line; do
    if [ $(((current_line - start) % increment)) -eq 0 ]; then
        # Execute the command for the current line
        python baysic/group_search.py --config_path=/home/miklaucn/hpc.toml --target.mp_id=$line
    fi

    ((current_line++))
done < "$file"
end=`date +%s`
runtime=$((end-start))
echo $runtime >> /home/miklaucn/times
