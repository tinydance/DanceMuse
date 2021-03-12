import os
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import numpy as np
from extractor import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--input_audio_dir', type=str, default='data/audio/ballet_1min/')
parser.add_argument('--test_dir', type=str, default='data/test/')
parser.add_argument('--sampling_rate', type=int, default=15360)
args = parser.parse_args()

if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)


extractor = FeatureExtractor()


def extract_acoustic_feature(input_audio_dir):
    print('---------- Extract features from raw audio ----------')
    musics = []
    # onset_beats = []
    audio_fnames = sorted(os.listdir(input_audio_dir))
   # audio_fnames = audio_fnames[:70]
    for audio_fname in audio_fnames:
        if (audio_fname[0]!='.'):
            audio_file = os.path.join(input_audio_dir, audio_fname)
            print(f'Process -> {audio_file}')
            ### load audio ###
            sr = args.sampling_rate
            loader = essentia.standard.MonoLoader(filename=audio_file, sampleRate=sr)
            audio = loader()
            audio = np.array(audio).T

            melspe_db = extractor.get_melspectrogram(audio, sr)
            mfcc = extractor.get_mfcc(melspe_db)
            mfcc_delta = extractor.get_mfcc_delta(mfcc)
            # mfcc_delta2 = get_mfcc_delta2(mfcc)

            audio_harmonic, audio_percussive = extractor.get_hpss(audio)
            # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
            # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
            chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr)
            # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

            onset_env = extractor.get_onset_strength(audio_percussive, sr)
            tempogram = extractor.get_tempogram(onset_env, sr)
            onset_beat = extractor.get_onset_beat(onset_env, sr)
            # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            # onset_beats.append(onset_beat)

            onset_env = onset_env.reshape(1, -1)

            feature = np.concatenate([
                 # melspe_db,
                mfcc,
                mfcc_delta,
                # mfcc_delta2,
                # harmonic_melspe_db,
                # percussive_melspe_db,
                # chroma_stft,
                chroma_cqt,
                onset_env,
                onset_beat,
                tempogram
            ], axis=0)

            feature = feature.transpose(1, 0)
            print(f'acoustic feature -> {feature.shape}')
            musics.append(feature.tolist())

    return musics

def save(args, musics):
    print('---------- Save to text file ----------')
    fnames = sorted(os.listdir(args.input_audio_dir))
    #fnames = fnames[:70]
    print(fnames)
    print(len(fnames))
    print(len(musics))
    assert len(fnames) == len(musics), 'alignment'

    # train_idx, valid_idx = split_data(fnames)
    # train_idx = sorted(train_idx)
    # print(f'train ids: {[fnames[idx] for idx in train_idx]}')
    # valid_idx = sorted(valid_idx)
    # print(f'valid ids: {[fnames[idx] for idx in valid_idx]}')

    print('---------- test data ----------')
    for idx,fname in enumerate(fnames):
        fn = os.path.splitext(fname)[0]
        with open(os.path.join(args.test_dir, f'{fn}.json'), 'w') as f:
            sample_dict = {
                'id': fnames[idx],
                'music_array': musics[idx],
            }
            json.dump(sample_dict, f)


if __name__ == '__main__':
    musics = extract_acoustic_feature(args.input_audio_dir)
    save(args, musics)
