# Minimally-Supervised Joint Learning of Event Volitionality and Subject Animacy Classification

## Requirements

- Python 3.8+
- torch: 1.8.1+
- transformers: 4.6.0+
- [Juman++](https://github.com/ku-nlp/jumanpp) (for Japanese)
- [KNP](https://github.com/ku-nlp/knp) (for Japanese)
- spacy (for English)
- and others (see [pyproject.toml](./pyproject.toml))

## Installation

Use [poetry](https://github.com/python-poetry/poetry).

```shell
poetry install

# Japanese
which jumanpp knp  # Ensure that Juman++ and KNP are installed

# English
poetry run python -m spacy download en_core_web_sm  # Download the model of en_core_web_sm
```

## Dataset Construction

### 1. Event extraction

Run the following commands.
This repository contains a very small portion of CC-100.
Visit [https://data.statmt.org/cc-100/](https://data.statmt.org/cc-100/) to download CC-100.

```shell
TARGET="ja"  # or "en"
mkdir ./data/extracted/$TARGET
make -k -f scripts/extract_event.mk \
  DATA_DIR=$(realpath ./data/cc-100/$TARGET) \
  RESULT_DIR=$(realpath ./data/extracted/$TARGET) \
  LANG=$TARGET \
  ROOT_DIR=$(pwd) \
  PYTHON=$(poetry run which python)
```

### 2. Dataset Construction

```shell
mkdir ./data/datasets/$TARGET
# Training
poetry run python src/create_training_dataset.py \
  --data_dir $(realpath ./data/extracted/$TARGET) \
  --output_dir $(realpath ./data/datasets/$TARGET)
# Evaluation
cp ./data/eval/$TARGET/*.jsonl ./data/datasets/$TARGET
gzip ./data/datasets/$TARGET/*.jsonl
```

When using Sampling and Occlusion (SOC), run the following commands in addition.

```shell
TARGET="en"
poetry run python src/assign_sample.py \
  --data_dir $(realpath ./data/datasets/$TARGET) \
  --model_name_or_path bert-base-cased

TARGET="ja"
# Download BERT pretrained by NICT (https://alaginrc.nict.go.jp/nict-bert/index.html)
wget https://alaginrc.nict.go.jp/nict-bert/NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip -O ./data/NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip
unzip ./data/NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip -d ./data
poetry run python src/assign_sample.py \
  --data_dir $(realpath ./data/datasets/$TARGET) \
  --model_name_or_path ./data/NICT_BERT-base_JapaneseWikipedia_32K_BPE
```

## Training

Run the following command.

```shell
TARGET="en"
poetry run python src/train.py \
  --vol {NONE|VANILLA|WR|SOC|ADA} \
  --ani {NONE|VANILLA|WR|SOC|ADA} \
  --data_dir ./data/datasets/$TARGET \
  --model_name_or_path bert-base-cased \
  --max_epochs 3

TARGET="ja"
poetry run python src/train.py \
  --vol {NONE|VANILLA|WR|SOC|ADA} \
  --ani {NONE|VANILLA|WR|SOC|ADA} \
  --data_dir ./data/datasets/$TARGET \
  --model_name_or_path ./data/NICT_BERT-base_JapaneseWikipedia_32K_BPE \
  --max_epochs 3
```
