#!/bin/bash
#
#SBATCH --job-name=DANCE_REVOLUTION
#SBATCH --output=/zooper2/tinydancer/DanceMuse/DR-NEW/v2/logs/test-log.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=8gb

echo "Starting slurm scipt"
echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"


# export CUDA_VISIBLE_DEVICES=3

export CUDA_HOME="/usr/local/cuda-10.0"
export PATH="$PATH:/usr/local/cuda-10.0/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64"
export PATH="$PATH:/usr/local/bin/conda"

work_dir="/zooper2/tinydancer/DanceMuse/test_v2"
model="/zooper2/tinydancer/DanceMuse/DR-NEW/v2/checkpoints/hiphop-continued/epoch_8000.pt"
input_dir="${work_dir}/test_audio"
output_dir="${work_dir}/test_output"
visualize_dir="${work_dir}/test_visualizations"

# Test
/zooper2/tinydancer/DanceRevolution/bin/python v2/test_audio_only.py --test_dir ${input_dir} \
--output_dir ${output_dir}} --model ${model} \
--visualize_dir ${visualize_dir}



