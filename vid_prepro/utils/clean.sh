#!/bin/bash

#######################
## Default variables ##
#######################
CLEAN_DIR="/zooper2/tinydancer/dances/"
CLEAN=$false
#################
## The script: ##
#################

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    	-i|--input_dir)
    	CLEAN_DIR="$2"
    	shift # past argument
    	shift # past value
    	;;
	-a|--all)
	CLEAN_ALL=$true
	shift # past argument
	;;
	*)    # unknown option
    	POSITIONAL+=("$1") # save it in an array for later
    	shift # past argument
    	;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Go to work dir
cd ${CLEAN_DIR}

rm -rf poses interpolated

if [ CLEAN_ALL ];then
	rm -rf audio edited_dances
fi
exit
