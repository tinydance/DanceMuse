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

model="/zooper2/tinydancer/DanceMuse/DR-NEW/v2/checkpoints/layers2_win500_schedule100_condition10_detach/epoch_4000.pt"
output_dir="/zooper2/tinydancer/DanceMuse/DR-NEW/hiphop-4k-outputs"
visualize_dir="/zooper2/tinydancer/DanceMuse/DR-NEW/hiphop-4k-visualizations"

# Test
python3 test.py --train_dir ../data/train_1min/hiphop --test_dir ../data/test_1min/hiphop \
--output_dir ${output_dir}} --model ${model} \
--visualize_dir ${visualize_dir}

files=$(ls ${visualize_dir})
for filename in $files
do
	ffmpeg -r 15 -i ${visualize_dir}/${filename}/frame%06d.jpg -vb 20M -vcodec mpeg4 \
	 -y ${visualize_dir}/${filename}.mp4
	echo "make video ${filename}"
done
