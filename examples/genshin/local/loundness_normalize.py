"""
音量响度归一化
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import librosa.filters
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]  # 双声道


def _pyloudnorm(audio, sample_rate):
    """
    https://github.com/csteinmetz1/pyloudnorm
    标准: EBU R128 Loudness Normalization and Permitted Maximum Level of Audio Signals
        The Programme Loudness Level shall be normalised to a Target Level of -23.0 LUFS
    """
    # measure the loudness first
    meter = pyln.Meter(sample_rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)

    # peak normalize audio to -1 dB
    # peak_normalized_audio = pyln.normalize.peak(data, -1.0)  # 貌似是峰值归一化

    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(audio, loudness, -23.0)
    return loudness_normalized_audio


def pyloundnorm_audio(audio, sample_rate, wavpath):
    sil_padsize = int(0.4 * sample_rate)
    # 在audio上添加 400ms 静音，因为pyloudnorm无法处理小于 400ms 的音频

    if len(audio.shape) > 1:
        print(f"{wavpath} is not mono!! skip")
        return
    duration = len(audio) / sample_rate
    if duration < 0.1:
        print(f"{wavpath} is too short, skip")
        return

    audio = np.concatenate([audio, np.zeros(sil_padsize)], 0)
    try:
        norm_audio = _pyloudnorm(audio, sample_rate)
    except Exception as e:
        print(f"norm {wavpath} failed")
        raise e
    norm_audio = norm_audio[:-sil_padsize]
    return norm_audio


def _pyloudnorm_file(wavpath, out_dir, input_dir):
    audio, sample_rate = sf.read(wavpath)
    norm_audio = pyloundnorm_audio(audio, sample_rate)

    outpath = out_dir / wavpath.relative_to(input_dir)
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(outpath), norm_audio, sample_rate)
    return outpath


def py_loudness_norm(input_dir: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    input_dir = Path(input_dir)

    # _pyloudnorm using concurrent futures with Process Pool
    futures = []
    with ProcessPoolExecutor(max_workers=20) as executor:
        for wavpath in tqdm(list(Path(input_dir).glob("**/*.wav")), desc="add to pool"):
            futures.append(
                executor.submit(_pyloudnorm_file, wavpath, out_dir, input_dir)
            )
        for future in tqdm(as_completed(futures), total=len(futures), desc="norm"):
            future.result()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("in_dir")
    parser.add_argument("out_dir")

    args = parser.parse_args()
    py_loudness_norm(args.in_dir, args.out_dir)
