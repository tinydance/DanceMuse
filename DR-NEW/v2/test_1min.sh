#! /bin/bash
model="/zooper2/tinydancer/DanceRevolution/trained_models/jpop_0405/epoch_6000.pt"
train_dir="/zooper2/tinydancer/DanceRevolution/data/train5j"
test_dir="/zooper2/tinydancer/DanceRevolution/data/valid5j"
output_dir="/zooper2/tinydancer/DanceMuse/DR-NEW/jpop-6k-0412-outputs"
visualize_dir="/zooper2/tinydancer/DanceMuse/DR-NEW/jpop-6k-0412-visualizations"

# Test
python3 test.py --train_dir ${train_dir} --test_dir ${test_dir} \
--output_dir ${output_dir}} --model ${model} \
--visualize_dir ${visualize_dir}

files=$(ls ${visualize_dir})
for filename in $files
do
	ffmpeg -r 15 -i ${visualize_dir}/${filename}/frame%06d.jpg -vb 20M -vcodec mpeg4 \
	 -y ${visualize_dir}/${filename}.mp4
	echo "make video ${filename}"
done

