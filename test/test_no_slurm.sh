#!/bin/bash

#######################
## Default variables ##
#######################
echo "<--Initializing Variables and Directories-->"
# input_dir="/zooper2/tinydancer/DanceRevolution/test_model/raw_audio/"
work_dir="/zooper2/tinydancer/DanceMuse/test"
duration="60"
fps="30"
model="/zooper2/tinydancer/DanceRevolution/trained_models/full_model/epoch_4000.pt"

#############################
## User Inputted Variables ##
#############################
while getopts i:m:p: flag
do
	case "${flag}" in
		i) input_dir=${OPTARG};;
		m) model=${OPTARG};;
		p) prefix=${OPTARG};;
	esac
done

#make dirs if they don't exist
mkdir -p edited_audio
mkdir -p test_audio
mkdir -p test_output
mkdir -p edited_output
edited_audio="${work_dir}/edited_audio"
test_audio="${work_dir}/test_audio"
test_output="${work_dir}/test_output"
edited_output="${work_dir}/edited_output"
image_dir="${work_dir}/test_output/outputs.test.images"

###################
## Preprocessing ##
###################
echo "<--Preprocessing Audio File-->"
#Iterate over the videos in the input directory
for audio in ${input_dir}/${prefix}*
do
	# Print progress
	echo "Processing ${audio}..."

	# Edit the video:
	#   *Don't overwrite mod files that exist
	#   *show warnings
	#   *force fps to  $fps
	#   *set video duration to $duration seconds
	#   *output into a filename_mod.mp4 file
	filename="${audio##*/}"
	new_audio="${edited_audio}/${filename%.*}.m4a"
	ffmpeg 	-n\
		-loglevel 24\
		-i "${audio}"\
		-r "${fps}"\
		-t "${duration}"\
		"${new_audio}"
done
echo "<--Extracting Audio Features-->"
/zooper2/tinydancer/DanceRevolution/bin/python prepro_test.py --input_audio_dir "${edited_audio}" \
	--test_dir "${test_audio}"

##############
# Run Model ##
##############
echo "<--Running Model-->"
json_dir="test_output/outputs.test.json"
/zooper2/tinydancer/DanceRevolution/bin/python test2.py --input_dir ${test_audio} \
	--model ${model} \
	--json_dir ${json_dir} \
	--image_dir ${image_dir} \
	--batch_size 1

####################
## Postprocessing ##
####################
echo "<--Merging Images into MP4-->"
# merge jpgs into mp4
dances=$(ls -d1 ${image_dir}/${prefix}* )
# echo "$dances"
for dance in ${dances}
do
	echo "Name of dance: $dance"
	filename="$(basename -s .mp4 $dance)"
	ffmpeg -r 15 -i ${dance}/frame%06d.jpg -vb 20M -vcodec mpeg4 -y ${test_output}/${filename}.mp4
	echo "Created ${test_output}/${filename}.mp4"
done

echo "<--Adding M4A to MP4-->"
# Recombine m4a's to mp4's
for dance in ${test_output}/${prefix}*.mp4
do
	filename="$(basename -s .mp4 $dance)"
	echo "${filename}"
	audio_filename=${filename%??}
	audio="${edited_audio}/$audio_filename.m4a"
	echo "${audio}"
	ffmpeg -i $dance -i $audio -c copy -map 0:v:0 -map 1:a:0 "${edited_output}/${filename}_with_audio.mp4"
done

echo "<--Testing Process Complete-->"

# Remove intermediary files
# rm ${edited_audio}/*.m4a
# rm ${test_audio}/*.json
# rm -r ${test_output}/outputs.test.images/*
# rm -r ${test_output}/outputs.test.json/*
# rm ${test_output}/*.mp4

# exit
