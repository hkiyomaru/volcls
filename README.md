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

Run the following command.
This repository contains a small sample of CC-100.
Visit [https://data.statmt.org/cc-100/](https://data.statmt.org/cc-100/) to download CC-100.

```shell
# Japanese
mkdir ./data/extracted/ja
make -k -f scripts/extract_event.mk \
  DATA_DIR=$(realpath ./data/cc-100/ja) \
  RESULT_DIR=$(realpath ./data/extracted/ja) \
  LANG=ja \
  ROOT_DIR=$(pwd) \
  PYTHON=$(poetry run which python)

# English
mkdir ./data/extracted/ja
make -k -f scripts/extract_event.mk \
  DATA_DIR=$(realpath ./data/cc-100/ja) \
  RESULT_DIR=$(realpath ./data/extracted/ja) \
  LANG=ja \
  ROOT_DIR=$(pwd) \
  PYTHON=$(poetry run which python)
```
