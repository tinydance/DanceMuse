#!/bin/bash
#
#SBATCH --job-name=DANCE_REVOLUTION
#SBATCH --output=/zooper2/tinydancer/DanceMuse/test/log.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=2gb

echo "Starting slurm scipt"
echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"


# export CUDA_VISIBLE_DEVICES=3

export CUDA_HOME="/usr/local/cuda-10.0"
export PATH="$PATH:/usr/local/cuda-10.0/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64"
export PATH="$PATH:/usr/local/bin/conda"

work_dir="/zooper2/tinydancer/DanceMuse/test"
image_dir="${work_dir}/test_output/0407-2k.outputs.test.images"
json_dir="${work_dir}/test_output/0407-2k.outputs.test.json"
input_dir="${work_dir}/test_audio"
output_dir="${work_dir}/test_output"
model="/zooper2/tinydancer/DanceRevolution/trained_models/jpop_0405/epoch_6000.pt"
	
/zooper2/tinydancer/DanceRevolution/bin/python "/zooper2/tinydancer/DanceRevolution/test2.py"  --input_dir ${input_dir} \
	--model ${model} \
	--json_dir ${json_dir} \
	--image_dir ${image_dir} \
	--batch_size 1



