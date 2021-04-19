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
	
# # ----------- condition step 10, sliding_window_size 100, lambda_v 0.01
# python3 train.py --train_dir ../data/train_1min --test_dir ../data/test_1min \
#                  --output_dir checkpoints/layers2_win100_schedule100_condition10_detach \
#                  --batch_size 32 --seq_len 900 --max_seq_len 4500 \
#                  --frame_emb_size 200 --d_pose_vec 50 --pose_emb_size 50 \
#                  --d_inner 1024 --n_layers 2 \
#                  --sliding_windown_size 100 --condition_step 10 --lambda_v 0.01

# ----------- condition step 10, sliding_window_size 500, lambda_v 0.01
python3 train.py --train_dir ../data/train_1min/hiphop --test_dir ../data/test_1min/hiphop \
                 --output_dir checkpoints/layers2_win500_schedule100_condition10_detach \
                 --batch_size 32 --seq_len 900 --max_seq_len 4500 \
                 --frame_emb_size 200 --d_pose_vec 50 --pose_emb_size 50 \
                 --d_inner 1024 --n_layers 2 \
                 --sliding_windown_size 500 --condition_step 10 --lambda_v 0.01 \
                 --epoch 4000 --save_per_epochs 1000


# # ----------- condition step 10, sliding_window_size 900, lambda_v 0.01
# python3 train.py --train_dir ../data/train_1min --test_dir ../data/test_1min \
#                  --output_dir checkpoints/layers2_win900_schedule100_condition10_detach \
#                  --batch_size 32 --seq_len 900 --max_seq_len 4500 \
#                  --frame_emb_size 200 --d_pose_vec 50 --pose_emb_size 50 \
#                  --d_inner 1024 --n_layers 2 \
#                  --sliding_windown_size 900 --condition_step 10 --lambda_v 0.01

