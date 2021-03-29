#!/bin/bash

# Calls interpolate_missing_keyjoints.py with the input path and ouput paths from the arguments. 

#######################
## Default variables ##
#######################
INPUT_DIR="/zooper2/tinydancer/dances/poses/"
OUTPUT_DIR="/zooper2/tinydancer/dances/"
INTERPOLATE_DIR="/zooper2/tinydancer/DanceRevolution/"


#################
## The script: ##
#################
#Argument Parsing
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



# Go to work dir
#cd ${OUTPUT_DIR}

#make dirs if they don't exist
mkdir -p interpolated
INTERPOLATED_DIR="${OUTPUT_DIR}interpolated/"


#cd  ${interpolation}
/zooper2/tinydancer/DanceRevolution/bin/python ${INTERPOLATE_DIR}interpolate_missing_keyjoints.py\
                --input_dir "$INPUT_DIR"\
                --output_dir "$INTERPOLATED_DIR"

exit
