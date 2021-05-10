edited_audio="edited_audio"
test_audio="test_audio"
test_output="test_output"
edited_ouput="edited_output"

# Remove intermediary files
rm ${edited_audio}/*
rm ${test_audio}/*
# rm -r ${test_output}/outputs.test.images/*
# rm -r ${test_output}/outputs.test.json/*
rm ${test_output}/*.mp4
rm ${edited_output}/*

