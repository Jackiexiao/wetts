# vits2-vocos-genshin

- vits2
- vocos vocoder: for fast inference

## genshin dataset
来源: https://github.com/AI-Hobbyist/Genshin_Datasets

- 语音数据集的所有权均归 米哈游 所有。
- 数据集仅供二次创作&模型训练，不得用于任何商业用途，不得用本仓库数据集训练的模型制作违反法律法规的内容，不得二次配布。如被发现滥用，将停止公开！

## install

```
pip install pyloudnorm soundfile tqdm
pip install WeTextProcessing
pip install pypinyin
```

if you use FunASR to recognize text, you need to install modelscope first.
```
pip install -U modelscope
```

## run
请先下载 genshin 数据
```
# 语音识别
bash run.sh --stage -1 --stop_stage -1
# 数据预处理
bash run.sh --stage 0 --stop_stage 0
# 训练
bash run.sh --stage 1 --stop_stage 1
```

## todo
- [ ] use wetts g2p instead of pypinyin 

## faq
- phones.list: 所有可能的音素
- phones.txt: 训练集中实际出现的音素和对应id
- train.txt: wavpath|speaker|phonemes|lang

## reference
https://github.com/p0p4k/vits2_pytorch
https://github.com/reppy4620/vocos_vits
