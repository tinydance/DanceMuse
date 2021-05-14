#!/bin/bash

# Read through the .mp4 in the input directory and return .m4a audio files for each.

#######################
## Default variables ##
#######################
INPUT_DIR="/zooper2/tinydancer/dances/raw_dances/"
OUTPUT_DIR="/zooper2/tinydancer/dances/"

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
    	*)    # unknown option
    	POSITIONAL+=("$1") # save it in an array for later
    	shift # past argument
    	;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# function to edit video and extract poses & audio
extract_audio(){
	video="$1"
        filename="${video##*/}"

	#get the .m4a audio file
	ffmpeg -i "${video}" -f mp3 -vn "${audio_dir}${filename%.*}.mp3"
}



cd ${OUTPUT_DIR}
#make dirs if they don't exist
mkdir -p audio
audio_dir="${OUTPUT_DIR}audio/"

# Iterate over the videos in the input directory and extract
for video in ${INPUT_DIR}*
do
	extract_audio "${video}"
done
exit
