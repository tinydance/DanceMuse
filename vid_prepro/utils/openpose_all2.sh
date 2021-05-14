#!/bin/bash

# Read through the .mp4 in the input directory, crop them to Sduration seconds and resample to  $fps fps. 
# Then call the $openpose with 1 person flag, and output .json and .avi for each of the edited videos
# the set of .json files for a clip go into their own dir, while the .avi are added to one folder. 


#######################
## Default variables ##
#######################
INPUT_DIR="/zooper2/tinydancer/dances/raw_dances/"
OUTPUT_DIR="/zooper2/tinydancer/dances/"
DURATION="60"
FPS="30"
openpose="/zooper2/tinydancer/openpose/build/examples/openpose/openpose.bin"
openpose_dir="/zooper2/tinydancer/openpose/"

#################
## The script: ##
#################

# Argument Parsing
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    	-i|--input_dir)
    	INPUT_DIR="$2"
    	shift # past argument
    	shift # past value
    	;;
    	-o|--output_dir)
    	OUTPUT_DIR="$2"
    	shift # past argument
    	shift # past value
    	;;
    	-d|--duration)
    	DURATION="$2"
    	shift # past argument
    	shift # past value
    	;;
	-r|-fps)
	FPS="$2"
	shift # past argument
	shift # past value
	;;
    	*)    # unknown option
    	POSITIONAL+=("$1") # save it in an array for later
    	shift # past argument
    	;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# function to edit video and extract poses & audio
extract_all(){
	video="$1"
 	# Edit the video:
        #   *Don't overwrite mod files that exist
        #   *show warnings
        #   *force fps to  $fps
        #   *set video duration to $duration seconds
        #   *output into a filename_mod.mp4 file
        filename="${video##*/}"

	# get the .m4a audio file
	# echo "filename"
	ffmpeg -i "${video}" -f mp3 -vn "${audio_dir}${filename%.*}.mp3"

	# create the output json folder
	json_folder="${pose_dir}${filename%.*}_poses/"
	mkdir -p "${json_folder}"

        # detect poses:
        cd ${openpose_dir}

	${openpose}     --video "${video}" \
                        --write_json "${json_folder}" \
                        --display 0 \
                        --number_people_max 1 \
			--face \
			--hand \
			--render_pose 0 \
			# Uncomment to save avi videos of the openpose outputs
#                      	--write_video "${videos_dir}${filename%.*}_poses.avi \
}



# Go to work dir
cd ${OUTPUT_DIR}

#make dirs if they don't exist
mkdir -p poses
mkdir -p audio
pose_dir="${OUTPUT_DIR}poses/"
audio_dir="${OUTPUT_DIR}audio/"

# Uncomment to create video directory
#mkdir -p videos
#videos_dir="${work_dir}videos/

# Iterate over the videos in the input directory and extract
for video in ${INPUT_DIR}*
do
	echo "${video}"
	extract_all "${video}"
done
exit
