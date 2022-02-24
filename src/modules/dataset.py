import logging
import random
import re
from abc import ABC
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from utils.utils import DataType, Event, Split, Target, read_event_file, read_seed_lex

logger = logging.getLogger(__file__)


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        data_dir: Path,
        model_name_or_path: str = "",
        max_seq_length: int = 128,
        target: Target = Target.VOL,
    ):
        self.data_dir = data_dir
        self.target = target
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.max_seq_length = max_seq_length

    def convert_text_to_features(self, text: str) -> Dict[str, torch.Tensor]:
        features = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        return {name: torch.LongTensor(feature) for name, feature in features.items()}

    def get_label(self, event: Event):
        return torch.Tensor(
            [event.vol_label if self.target == Target.VOL else event.ani_label]
        )


class TrainingDataset(BaseDataset):
    def __init__(
        self,
        seed_lex: Optional[str] = None,
        limit_train_data: Optional[int] = None,
        num_soc_samples: int = 3,
        *args,
        **kwargs,
    ):
        super(TrainingDataset, self).__init__(*args, **kwargs)
        seed_words = read_seed_lex(Path(seed_lex)) if seed_lex else None

        self.num_soc_samples = num_soc_samples

        def filter_fn(event: Event) -> bool:
            if seed_words is None:
                return True
            if self.target == Target.VOL:
                return event.vol_phrase in {w.text for w in seed_words}
            else:
                return event.ani_phrase in {w.text for w in seed_words}

        self.events = list(
            filter(
                filter_fn,
                read_event_file(
                    self.data_dir
                    / "{}.{}.{}.jsonl.gz".format(
                        self.target.name.lower(),
                        DataType.LABELED.name.lower(),
                        Split.TRAIN.name.lower(),
                    )
                ),
            )
        )
        self.events_unlabeled = read_event_file(
            self.data_dir
            / "{}.{}.{}.jsonl.gz".format(
                self.target.name.lower(),
                DataType.UNLABELED.name.lower(),
                Split.TRAIN.name.lower(),
            )
        )

        if limit_train_data is not None and limit_train_data > 0:
            size = min(len(self.events), limit_train_data)
            logger.info(f"Limit training data: {len(self.events):,} -> {size:,}")
            self.events = random.sample(self.events, size)
            self.events_unlabeled = random.sample(self.events_unlabeled, size)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx: int):
        event = self.events[idx]
        features = self.convert_text_to_features(event.tokenized_text)
        features["labels"] = self.get_label(event)

        # For WR.
        seed_word = event.vol_phrase if self.target == Target.VOL else event.ani_phrase
        tokenized_text_wr = self.remove_word(event.tokenized_text, seed_word)
        features_wr = self.convert_text_to_features(tokenized_text_wr)
        for name, feature in features_wr.items():
            features[name + "_wr"] = feature

        # For SOC.
        for i, tokenized_sampled_text in enumerate(
            random.sample(event.tokenized_sampled_text, self.num_soc_samples)
        ):
            features_s = self.convert_text_to_features(tokenized_sampled_text)
            for name, feature in features_s.items():
                features[name + f"_soc_s_{i}"] = feature
            tokenized_sampled_text_soc = self.occlude_word(
                tokenized_sampled_text, seed_word
            )
            features_soc = self.convert_text_to_features(tokenized_sampled_text_soc)
            for name, feature in features_soc.items():
                features[name + f"_soc_soc_{i}"] = feature

        # For ADA & CON.
        event_unlabeled = self.events_unlabeled[idx]
        features_unlabeled = self.convert_text_to_features(
            event_unlabeled.tokenized_text
        )
        for name, feature in features_unlabeled.items():
            features[name + "_unlabeled"] = feature

        return features

    @staticmethod
    def remove_word(text: str, word: str):
        try:
            return re.sub(r"\s?".join(word) + r"\s", "", text)
        except Exception as e:
            logger.warning(e)
            return text

    def occlude_word(self, text: str, word: str):
        try:
            return re.sub(
                r"\s?".join(word) + r"\s", self.tokenizer.pad_token + " ", text
            )
        except Exception as e:
            logger.warning(e)
            return text


class EvaluationDataset(BaseDataset):
    def __init__(self, data_type: DataType, split: DataType, *args, **kwargs):
        super(EvaluationDataset, self).__init__(*args, **kwargs)
        assert split in {Split.VAL, Split.TEST}
        if data_type == DataType.ALL:
            data_types = (DataType.LABELED, DataType.UNLABELED)
        else:
            data_types = (data_type,)
        events = []
        for data_type in data_types:
            data_path = self.data_dir / "{}.{}.{}.jsonl.gz".format(
                self.target.name.lower(),
                data_type.name.lower(),
                split.name.lower(),
            )
            events.extend(read_event_file(data_path))
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx: int):
        event = self.events[idx]
        features = self.convert_text_to_features(event.tokenized_text)
        features["labels"] = self.get_label(event)
        return features
