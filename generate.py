import argparse
import os
import json

import torch
import numpy as np
import glob


from model import Vocoder
from utils import load_wav, save_wav, melspectrogram


def gen_from_mel(model, mel, output):
    assert mel.shape[1] == 80, 'Input mel shape is invalid.'
    assert output.endswith('.wav')
    mel = torch.FloatTensor(mel).unsqueeze(0).to(device)
    waveform = model.generate(mel)
    save_wav(output, waveform, params["preprocessing"]["sample_rate"])
    #librosa.output.write_wav(output, waveform, sr=16000)

def gen_from_wav(model, wav, output):
    wav = load_wav(wav, params["preprocessing"]["sample_rate"], trim=False)
    utterance_id = os.path.basename(args.input).split(".")[0]
    wav = wav / np.abs(wav).max() * 0.999
    mel = melspectrogram(wav, sample_rate=params["preprocessing"]["sample_rate"],
                         preemph=params["preprocessing"]["preemph"],
                         num_mels=params["preprocessing"]["num_mels"],
                         num_fft=params["preprocessing"]["num_fft"],
                         min_level_db=params["preprocessing"]["min_level_db"],
                         ref_level_db=params["preprocessing"]["ref_level_db"],
                         hop_length=params["preprocessing"]["hop_length"],
                         fmin=params["preprocessing"]["fmin"],
                         fmax=params["preprocessing"]["fmax"])
    gen_from_mel(model, mel, output)

def gen_from_npy(model, npy, output):
    mel = np.load(npy)#.reshape((80, -1))
    if mel.shape[1] != 80:
        mel = mel.T
    gen_from_mel(model, mel, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help="Checkpoint path to resume")
    parser.add_argument('--outdir', '-o', type=str, help='output dir', default="./generated")
    parser.add_argument('--input', '-i', type=str)
    args = parser.parse_args()
    with open("config.json") as f:
        params = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Vocoder(mel_channels=params["preprocessing"]["num_mels"],
                    conditioning_channels=params["vocoder"]["conditioning_channels"],
                    embedding_dim=params["vocoder"]["embedding_dim"],
                    rnn_channels=params["vocoder"]["rnn_channels"],
                    fc_channels=params["vocoder"]["fc_channels"],
                    bits=params["preprocessing"]["bits"],
                    hop_length=params["preprocessing"]["hop_length"])
    model.to(device)

    print(f'Load checkpoint from: {args.resume}:')
    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model_step = checkpoint["step"]




    if os.path.isdir(args.input):
        wavs = glob.glob(os.path.join(args.input, '*.wav'))
        npys = glob.glob(os.path.join(args.input, '*.npy'))
        assert args.input != args.outdir, '[Error] Input and output dir should be different.'
        assert not os.path.isfile(args.outdir), '[Error] Output should be a directory.'
        assert len(wavs) == 0 or len(npys) == 0, f'[Error] Both .wav and .npy exist in {args.input}.'
        assert len(wavs) != 0 or len(npys) != 0, f'[Error] No .wav or .npy exist in {args.input}.'
        os.makedirs(args.outdir, exist_ok=True)
        for wav in wavs:
            basename = os.path.basename(wav)
            basename_out = f'gen_{basename}_model_steps_{model_step}.wav'
            gen_from_wav(model, wav, os.path.join(args.outdir, basename_out))
        for npy in npys:
            basename = os.path.basename(npy).split('.npy')[0]
            basename_out = f'gen_{basename}_model_steps_{model_step}.wav'
            gen_from_npy(model, npy, os.path.join(args.outdir, basename_out))
    elif args.input.endswith('.npy'):
        basename = os.path.basename(args.input).split('.npy')[0]
        basename_out = f'gen_{basename}_model_steps_{model_step}.wav'
        gen_from_npy(model, args.input, os.path.join(args.outdir, basename_out))
    elif args.input.endswith('.wav'):
        basename = os.path.basename(args.input).split('.wav')[0]
        basename_out = f'gen_{basename}_model_steps_{model_step}.wav'
        gen_from_wav(model, args.input, os.path.join(args.outdir, basename_out))
    else:
        print(f'[Error] Invalid input "{args.input}".')
