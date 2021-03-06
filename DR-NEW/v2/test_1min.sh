#! /bin/bash
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

