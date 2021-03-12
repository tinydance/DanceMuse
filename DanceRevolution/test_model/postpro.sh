work_dir="/zooper2/tinydancer/DanceRevolution/"
edited_audio="${work_dir}test_model/edited_audio"
test_audio="${work_dir}test_model/test_audio"
image_dir="${work_dir}test_model/test_output/outputs.test.images"
test_output="${work_dir}test_model/test_output"
edited_output="${work_dir}test_model/edited_output"
prefix=$1
####################
## Postprocessing ##
####################

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

# Recombine m4a's to mp4's
for dance in ${test_output}/${prefix}*.mp4
do
	filename="$(basename -s .mp4 $dance)"
	audio="${edited_audio}/$filename.m4a"
	ffmpeg -i $dance -i $audio -c copy -map 0:v:0 -map 1:a:0 "${edited_output}/${filename}_with_audio.mp4"
done