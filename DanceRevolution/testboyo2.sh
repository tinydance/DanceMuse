#!/bin/bash
#
#SBATCH --job-name=DANCE_REVOLUTION
#SBATCH --output=/zooper2/tinydancer/DanceRevolution/logs/0223_ballet_3000_test.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=2gb

#This is a sample SLURM job script
#It has a job name called JOB_NAME
#It will save it's stdout to /zooper1/USERNAME/output.txt
#It will have access to 2 GPUs and 1GB of RAM for a maximum of 1 minute
#The minimum allocatable time is 1 minute

echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"

# export CUDA_VISIBLE_DEVICES=3

export CUDA_HOME="/usr/local/cuda-10.0"
export PATH="$PATH:/usr/local/cuda-10.0/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64"
export PATH="$PATH:/usr/local/bin/conda"

image_dir=output_ballet/outputs.test_1990.images
	
/zooper2/tinydancer/DanceRevolution/bin/python test2.py --input_dir data/valid_1min \
	--model trained_models/ballet_only/epoch_1990.pt \
	--json_dir output_ballet/outputs.test_1990.json \
	--image_dir ${image_dir} \
	--batch_size 1

dances=$(ls ${image_dir})
for dance in ${dances}:
do
	ffmpeg -r 15 -i ${image_dir}/${dance}/frame%06d.jpg -vb 20M -vcodec mpeg4 -y ${image_dir}/${dance}.mp4
	echo "make video {images/${dance}.mp4}"
done

