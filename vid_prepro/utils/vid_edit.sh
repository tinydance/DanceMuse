#!/bin/bash

# Read through the .mp4 in the input directory, crop them to $duration seconds and resample to  $fps fps.


#######################
## Default variables ##
#######################
INPUT_DIR="/zooper2/tinydancer/dances/raw_dances/"
OUTPUT_DIR="/zooper2/tinydancer/dances/"
DURATION="60"
FSP="15"

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
	-r|-fsp)
	FSP="$2"
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
edit_video(){
	video="$1"
 	# Edit the video:
        #   *Don't overwrite mod files that exist
        #   *show warnings
        #   *force fps to  $fps
        #   *set video duration to $duration seconds
        #   *output into a filename_mod.mp4 file
        filename="${video##*/}"
        ffmpeg -i "${video}" -filter:v fps=fps=30 "${filename}_30fps.mp4"
        
        original_duration=$(ffprobe -i "${filename}_30fps.mp4" -show_entries format=duration -v quiet -of csv="p=0")
        num_chunks=$((${original_duration} % ${DURATION}))
        start_time=0
        for i in{1..${num_chunks}}
        do
            edited_vid="${edited_dir}${filename%.*}_m.avi"
            ffmpeg -i "${filename}_30fps.mp4" -ss "${start_time}" -t "${DURATION}" -c copy "${edited_vid}"
            start_time=$((${start_time} + ${DURATION}))
        done
        rm "${filename}_30fps.mp4"
        # ffmpeg  -n\
        #         -loglevel 24\
        #         -i "${video}"\
        #         -r "${FSP}"\
        #         -t "${DURATION}"\
        #         "${edited_vid}"
}

# Go to work dir
cd ${work_dir}

#make dirs if they don't exist
mkdir -p edited_dances
edited_dir="${OUTPUT_DIR}edited_dances/"

# Iterate over the videos in the input directory and extract
for video in ${INPUT_DIR}*.mp4
do
	edit_video "${video}"
done
exit
