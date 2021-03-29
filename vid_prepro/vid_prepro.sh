#!/bin/bash

###############
# Defualt Variables
DURATION="60"
FPS="15"
CLEAN=$false
OUTPUT_DIR="$(pwd)/"
INPUT_DIR="/zooper2/tinydancer/dances/raw_dances/"
UTILS_DIR="/zooper2/tinydancer/dances/utils/"

#Fix so end of input dirs always end in a slash
ammend_end(){
	strng="$1"
	if [ "${input: -1}" != "/" ]; then
		strng="${strng}/"
	else
		strng="${strng}"
	fi
}

# Argument Parsing:
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
	-i|--input_dir)
	ammend_end "$2"
    	INPUT_DIR="${strng}"
    	shift # past argument
    	shift # past value
    	;;
    	-o|--output_dir)
	ammend_end "$2"
    	OUTPUT_DIR="${strng}"
    	shift # past argument
    	shift # past value
    	;;
	-d|--duration)
	DURATION="$2"
	shift # past argument
	shift # past value
	;;
	-r|--fps)
	FPS="$2"
	shift # past argument
	shift # past value
	;;
    	-c|--clean)
    	CLEAN=$true
    	shift # past argument
    	;;
	--help)
	echo "Usage:"
	echo "$0 [OPTION]... [FILE/VALUE]..." 
	echo "Available Options"
	echo "-i, --input_dir  		input path to directory with raw videos"
	echo "-o, --output_dir  		desired output path. Defaults to current direcotry"
	echo "-d, --duration  		desired duration of video clip. Defaults to one minute"
	echo "-r, --fps			desired input fps. Defaults to 30 fps"
	echo "-c, --clean			whether or not to clean the output directory before processing. Defaults to not clean"
    	echo "--help				displays this help and exits"
	exit
	;;
	*)    # unknown option
    	POSITIONAL+=("$1") # save it in an array for later
    	shift # past argument
    	;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Print out the specs
echo "Preprocessing Videos with the following specifications:"
echo "Input Directory: ${INPUT_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Video Duration: ${DURATION}"
echo "Video fsp: ${FPS}"


mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}
# remove old specs if exist
rm -f output_specs.txtx



# If cleaning is enabled, clean
if [ ${CLEAN} ]; then
	echo "Cleaning directories"
	${UTILS_DIR}clean.sh -c ${OUTPUT_DIR}

fi
# create and run the pose extraction slurm file
cp ${UTILS_DIR}empty_slurm.sh slurm.sh
echo  "${UTILS_DIR}openpose_all.sh -i ${INPUT_DIR} -o ${OUTPUT_DIR} -d ${DURATION} -r ${FPS}" >> slurm.sh
sbatch -W slurm.sh
wait
rm slurm.sh

# interpolate the missing keyjoints
${UTILS_DIR}interpolate.sh -i "${OUTPUT_DIR}poses/" -o "${OUTPUT_DIR}" 

# Save the specs used to generate the outputs
touch output_specs.txt 
echo "Specifications for the outputs:" >> output_specs.txt
echo "Input Dir: ${INPUT_DIR}" >> output_specs.txt
echo "Video Durations: ${DURATION}" >> output_specs.txt
echo "Video FPS: ${FPS}" >> output_specs.txt

exit
