import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm


def convert_to_list(args):
    data_dir = Path(args.data_dir)

    output_list = []
    print("Find ASR.json ......")
    for asr_path in tqdm(list(data_dir.glob("**/ASR.json"))):
        with open(asr_path, "r") as f:
            data = json.load(f)

        for wavpath, values in data.items():
            wavpath = Path(wavpath)
            text = values.get("text", "")
            speaker = wavpath.parent.name
            wavname = wavpath.name
            actual_wavpath = data_dir / speaker / wavname
            if actual_wavpath.exists():
                output_list.append(f"{actual_wavpath}|{speaker}|{args.lang}|{text}")
            else:
                print(f"File not found: {actual_wavpath}")

    with open(args.outfile, "w") as f:
        f.write("\n".join(output_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON file to list")
    parser.add_argument(
        "--outfile", type=str, help="Output file",
    )
    parser.add_argument(
        "--data_dir", type=str,
    )
    parser.add_argument(
        "--lang", type=str, default="zh", choices=["zh", "en", "jp", "ko"]
    )

    args = parser.parse_args()

    convert_to_list(args)
