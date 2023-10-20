#!/usr/bin/env bash

# Copyright 2022 Jie Chen
# Copyright 2022 Binbin Zhang(binbzha@qq.com)

[ -f path.sh ] && . path.sh

export CUDA_VISIBLE_DEVICES="5"

# download dataset first
# dataset url : https://github.com/AI-Hobbyist/Genshin_Datasets
stage=0
stop_stage=3

raw_dataset_dir=path/to/dataset/Genshin-Chinese

# training dir
dir=exp/vits2_vocos_genshin
config=configs/vits2_vocos.json

# data_dir / dump_dir
data=~/data/dump/Genshin-Chinese
test_audio=test_audio

. tools/parse_options.sh || exit 1;

echo "======================"
echo "stage: ${stage}"
echo "stop_stage: ${stop_stage}"
echo "======================"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  mkdir -p $data
  python local/asr.py --data_dir $raw_dataset_dir 
fi

# keep 0.5h ~ 2h (1800s ~ 7200s) data for each speaker
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  python local/get_datalist.py --data_dir $raw_dataset_dir --lang zh --outfile $data/raw.txt
  python local/audio_preprocess.py --datalist $data/raw.txt --dump_dir $data --new_datalist $data/filter.txt -s 24000 \
    --min_duration 0.5 --max_duration 12.0 --min_duration_speaker 1800 --max_duration_speaker 7200
  python tools/gen_pinyin_lexicon.py \
    --with-zero-initial --with-tone --with-r \
    $data/lexicon.txt \
    $data/phones.list
  python local/prepare_data.py \
    --datalist $data/filter.txt \
    --dump_dir $data \
    --lexicon_path $data/lexicon.txt
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  export MASTER_ADDR=localhost
  export MASTER_PORT=10098
  python vits/train.py -c $config -m $dir \
    --train_data $data/train.txt \
    --val_data $data/val.txt \
    --speaker_table $data/speaker.txt \
    --phone_table $data/phones.txt
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  mkdir -p $test_audio
  python vits/inference.py --cfg $config \
    --speaker_table $data/speaker.txt \
    --phone_table $data/phones.txt \
    --checkpoint $dir/G_90000.pth \
    --test_file $data/test.txt \
    --outdir $test_audio
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  mkdir -p $test_audio
  python vits/export_onnx.py --cfg $config \
    --speaker_table $data/speaker.txt \
    --phone_table $data/phones.txt \
    --checkpoint $dir/G_90000.pth \
    --onnx_model $dir/G_90000.onnx

  python vits/inference_onnx.py --cfg $config \
    --speaker_table $data/speaker.txt \
    --phone_table $data/phones.txt \
    --onnx_model $dir/G_90000.onnx \
    --test_file $data/test.txt \
    --outdir $test_audio
fi
