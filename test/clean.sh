edited_audio="edited_audio"
test_audio="test_audio"
test_output="test_output"

# Remove intermediary files
rm ${edited_audio}/*.m4a
rm ${test_audio}/*.m4a
# rm -r ${test_output}/outputs.test.images/*
# rm -r ${test_output}/outputs.test.json/*
rm ${test_output}/*.mp4

