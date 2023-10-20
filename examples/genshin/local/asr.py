import argparse
import os
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def main(data_dir, lang):
    if lang == "zh":
        model_name = "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    elif lang == "en":
        model_name = "damo/speech_paraformer_asr-en-16k-vocab4199-pytorch"
    else:
        raise ValueError(f"Unsupported language {lang}")

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=model_name,
    )

    for spk in os.listdir(data_dir):
        spk_dir = os.path.join(data_dir, spk)
        out_dir = os.path.join(spk_dir, "ASR.json")
        if os.path.exists(out_dir):
            print(f"speaker {spk} has already been processed.")
            continue
        js = {}
        for audio in os.listdir(spk_dir):
            if not audio.endswith(".wav"):
                continue

            audio_dir = os.path.join(spk_dir, audio)
            try:
                rec_result = inference_pipeline(audio_in=audio_dir)
                js[audio_dir] = rec_result
            except Exception as e:
                print(f"Error encountered {e}, at {audio_dir}")

        with open(out_dir, "w", encoding="utf-8") as f:
            json.dump(js, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fun ASR")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Data directory containser speaker_1/xx.wav, speaker_2/xx.wav",
    )
    parser.add_argument("--lang", type=str, default="zh")

    args = parser.parse_args()

    main(args.data_dir, args.lang)
