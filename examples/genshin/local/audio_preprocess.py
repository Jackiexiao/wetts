"""
音频预处理

- 音频长度筛选
- 音量归一化
- 重采样
- 按说话人划分
- 按说话人音频长度筛选
"""

import argparse
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List
import os

from loundness_normalize import pyloundnorm_audio


def resample_wav(func_args):
    wavpath, out_wavpath, target_rate, min_duration, max_duration = func_args
    if Path(wavpath).exists():
        out_wavpath.parent.mkdir(parents=True, exist_ok=True)
        audio, sr = librosa.load(wavpath, sr=target_rate)
        if len(audio.shape) > 1:
            audio = audio[0]
        duration = len(audio) / sr

        if min_duration and duration < min_duration:
            print(
                f"Skip audio {wavpath} due to duration {duration:.2f}s < min_duration {min_duration:.2f}s"
            )
            return
        if max_duration and duration > max_duration:
            print(
                f"Skip audio {wavpath} due to duration {duration:.2f}s > max_duration {max_duration:.2f}s"
            )
            return

        audio = pyloundnorm_audio(audio, sr, wavpath)

        sf.write(out_wavpath, audio, target_rate)
    else:
        print(f"{wavpath} does not exist")


def get_duration(wavpath):
    if Path(wavpath).exists():
        try:
            duration = librosa.get_duration(path=wavpath)
        except TypeError:
            duration = librosa.get_duration(
                filename=wavpath
            )  # use filename=wavpath when in librosa < 0.10
        return duration
    else:
        return 0


def resample_wav_multi_process(
    wavpaths: List[str],
    out_wavpaths: List[str],
    target_sample_rate=24000,
    n_job=20,
    min_duration=None,
    max_duration=None,
    min_duration_speaker=None,
    max_duration_speaker=None,
):
    wavs_len = len(wavpaths)
    func_args = list(
        zip(
            wavpaths,
            out_wavpaths,
            wavs_len * [target_sample_rate],
            wavs_len * [min_duration],
            wavs_len * [max_duration],
        )
    )
    with Pool(processes=n_job) as p:
        with tqdm(total=len(wavpaths), desc="resample & loudnorm") as pbar:
            for i, _ in enumerate(p.imap_unordered(resample_wav, func_args)):
                pbar.update()

    out_wavpaths = [x for x in out_wavpaths if Path(x).exists()]

    if min_duration_speaker or max_duration_speaker:
        speaker_wav_dict = {}
        for i, wavpath in enumerate(out_wavpaths):
            speaker = os.path.basename(os.path.dirname(wavpath))
            if speaker not in speaker_wav_dict:
                speaker_wav_dict[speaker] = []
            speaker_wav_dict[speaker].append(wavpath)

        print(f"Total speakers: {len(speaker_wav_dict)}")

        speaker_duration = {}

        for speaker, speaker_wavpaths in tqdm(
            speaker_wav_dict.items(),
            desc=f"Check duration speaker : {min_duration_speaker} < duration < {max_duration_speaker}",
        ):
            total_duration = 0
            for wavpath in speaker_wavpaths:
                duration = get_duration(wavpath)
                total_duration += duration

            if min_duration_speaker and total_duration < min_duration_speaker:
                for wavpath in speaker_wavpaths:
                    if Path(wavpath).exists():
                        os.remove(wavpath)
                os.rmdir(os.path.dirname(wavpath))
                print(
                    f"Removed speaker {speaker} due to total duration {total_duration:.2f}s < min_duration_speaker {min_duration_speaker:.2f}s"
                )
                continue

            if max_duration_speaker and total_duration > max_duration_speaker:
                speaker_wavpaths.sort(key=lambda x: get_duration(x), reverse=True)
                for wavpath in speaker_wavpaths:
                    duration = get_duration(wavpath)
                    if total_duration - duration > max_duration_speaker:
                        if Path(wavpath).exists():
                            os.remove(wavpath)
                        total_duration -= duration
                        print(
                            f"Removed audio {wavpath} due to total duration {total_duration:.2f}s > max_duration_speaker {max_duration_speaker:.2f}s"
                        )
                    else:
                        break
            speaker_duration[speaker] = total_duration
        # save speaker_duration
        # with open("speaker_duration.txt", "w") as f:
        #     for speaker, duration in speaker_duration.items():
        #         info = f"{speaker} {duration:.2f}\n"
        #         print(info.strip())
        #         f.write(info)


def resample_wav_dir(
    datalist,
    dump_dir,
    target_sample_rate=24000,
    n_job=20,
    min_duration=None,
    max_duration=None,
    min_duration_speaker=None,
    max_duration_speaker=None,
):
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)

    if datalist:
        with open(datalist, "r") as f:
            lines = f.readlines()
            wavpaths = [Path(x.strip().split("|", 1)[0]) for x in lines]
            out_wavpaths = [dump_dir / x.parent.name / x.name for x in wavpaths]

            other_infos = [x.strip().split("|", 1)[1] for x in lines]

    else:
        raise ValueError("datalist and in_dir should not be both None")

    resample_wav_multi_process(
        wavpaths,
        out_wavpaths,
        target_sample_rate,
        n_job,
        min_duration,
        max_duration,
        min_duration_speaker,
        max_duration_speaker,
    )

    with open(args.new_datalist, "w") as f:
        for out_wavpath, other_info in zip(out_wavpaths, other_infos):
            if Path(out_wavpath).exists():
                f.write(f"{out_wavpath}|{other_info}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", type=str, help="音频路径列表")
    parser.add_argument("--dump_dir", type=str )
    parser.add_argument("--new_datalist", type=str )

    parser.add_argument("-s", "--sample_rate", default=24000, type=int)
    parser.add_argument("-n", "--n_job", default=max(1, cpu_count() // 2), type=int)
    parser.add_argument(
        "--min_duration",
        default=0.5,
        type=float,
        help="min duration (seconds) to filter audio",
    )
    parser.add_argument(
        "--max_duration",
        default=15.0,
        type=float,
        help="max duration (seconds) to filter audio",
    )
    parser.add_argument(
        "--min_duration_speaker",
        default=None,
        type=float,
        help="min_duration_speaker (seconds) to filter speaker",
    )
    parser.add_argument(
        "--max_duration_speaker",
        default=None,
        type=float,
        help="max_duration_speaker (seconds) to remove extra audio of speaker for data blance",
    )
    args = parser.parse_args()
    print(args)

    resample_wav_dir(
        args.datalist,
        args.dump_dir,
        args.sample_rate,
        args.n_job,
        args.min_duration,
        args.max_duration,
        args.min_duration_speaker,
        args.max_duration_speaker,
    )
