"""Assign sampled sentences to each sentence in the modules datasets.

Usage:
    env CUDA_VISIBLE_DEVICES=0 python src/assign_sample.py \
        --data_dir ... \
        --model_name_or_path ...
"""
import argparse
import logging
import random
import re
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizerFast

from utils.utils import DataType, Split, Target, read_event_file, save_event_file

logger = logging.getLogger(__file__)

special_token_ids: List[int] = []


def set_special_token_ids(tokenizer: BertTokenizerFast) -> None:
    global special_token_ids
    for special_token in tokenizer.special_tokens_map.values():
        special_token_id = tokenizer.get_vocab()[special_token]
        if special_token_id not in special_token_ids:
            special_token_ids.append(special_token_id)
    special_token_ids.sort()


class MLMDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        model_name_or_path: str = "",
        max_seq_length: int = 64,
        target: Target = Target.VOL,
    ):
        self.data_dir = data_dir
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.max_seq_length = max_seq_length
        self.target = target

        self.sents = read_event_file(
            self.data_dir
            / "{}.{}.{}.jsonl.gz".format(
                self.target.name.lower(),
                DataType.LABELED.name.lower(),
                Split.TRAIN.name.lower(),
            )
        )

        for sent in self.sents:
            sent.tokenized_sampled_text = []
        self.add_sampled_sents([s.tokenized_text for s in self.sents])

        set_special_token_ids(self.tokenizer)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx: int):
        sent = self.sents[idx]
        phrase = sent.vol_phrase if self.target == Target.VOL else sent.ani_phrase
        features = self.convert_text_to_features(
            sent.tokenized_sampled_text[-1], phrase
        )
        return features

    def convert_text_to_features(
        self, text: str, phrase: str
    ) -> Dict[str, torch.Tensor]:
        try:
            text = re.sub(r"\s?".join(phrase), self.tokenizer.sep_token, text)
        except Exception as e:
            logger.error(f"{e}; input was '{text} (phrase: {phrase})'")
        features = self.tokenizer(text)
        phrase_features = self.tokenizer(phrase, add_special_tokens=False)

        # get ID to mask
        sep_id = features["input_ids"].index(self.tokenizer.sep_token_id)
        if sep_id == len(features["input_ids"]) - 1:
            mask_id = -1
        else:
            cand_ids = [
                i
                for i, id_ in enumerate(features["input_ids"])
                if i < self.max_seq_length and id_ not in special_token_ids
            ]
            mask_id = random.choice(cand_ids) if cand_ids else -1

            # replace ID
            if mask_id >= 0:
                features["input_ids"][mask_id] = self.tokenizer.mask_token_id
            features["input_ids"] = (
                features["input_ids"][:sep_id]
                + phrase_features["input_ids"]
                + features["input_ids"][sep_id + 1 :]
            )
            features["token_type_ids"] = (
                features["token_type_ids"][:sep_id]
                + phrase_features["token_type_ids"]
                + features["token_type_ids"][sep_id + 1 :]
            )
            features["attention_mask"] = (
                features["attention_mask"][:sep_id]
                + phrase_features["attention_mask"]
                + features["attention_mask"][sep_id + 1 :]
            )

        # Truncate
        features["input_ids"] = features["input_ids"][: self.max_seq_length]
        features["token_type_ids"] = features["token_type_ids"][: self.max_seq_length]
        features["attention_mask"] = features["attention_mask"][: self.max_seq_length]

        num_pad_tokens = self.max_seq_length - len(features["input_ids"])
        features["input_ids"] += [self.tokenizer.pad_token_id] * num_pad_tokens
        features["token_type_ids"] += [0] * num_pad_tokens
        features["attention_mask"] += [0] * num_pad_tokens

        features["mask_id"] = [mask_id]

        return {name: torch.LongTensor(feature) for name, feature in features.items()}

    def add_sampled_sents(self, sampled_sents: List[str]):
        for sent, sampled_sent in zip(self.sents, sampled_sents):
            sent.tokenized_sampled_text.append(sampled_sent)

    def drop_first_n_sampled_sents(self, n: int):
        for sent in self.sents:
            sent.tokenized_sampled_text = sent.tokenized_sampled_text[n:]


def get_bert(model_name_or_path: str):
    logger.info("Load BERT.")
    bert = BertForMaskedLM.from_pretrained(model_name_or_path)
    bert.eval()
    return bert


def get_dataset(
    data_dir: str, model_name_or_path: str, max_seq_length: int, target: Target
):
    logger.info(f"Load the dataset for {target}.")
    return MLMDataset(
        data_dir=Path(data_dir),
        model_name_or_path=model_name_or_path,
        max_seq_length=max_seq_length,
        target=target,
    )


def sample_sentences(
    bert: BertForMaskedLM,
    loader: DataLoader,
    tokenizer: BertTokenizerFast,
    device: str,
):
    sampled_sents = []
    for batch in tqdm(loader, desc="Progress"):
        batch = {k: v.to(device) for k, v in batch.items()}

        mask_id = batch["mask_id"].view(-1)
        batch_size = len(batch["mask_id"])

        next_input_ids = batch["input_ids"].clone()
        if (mask_id != -1).sum().item() != 0:
            with torch.no_grad():
                outputs = bert(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                logits[:, :, special_token_ids] -= 128.0
                indices_with_mask_0 = torch.arange(batch_size, device=device)[
                    mask_id != -1
                ]
                indices_with_mask_1 = mask_id[mask_id != -1]
                next_input_ids[indices_with_mask_0, indices_with_mask_1] = Categorical(
                    logits=logits[indices_with_mask_0, indices_with_mask_1]
                ).sample()
        sampled_sents.extend(
            tokenizer.batch_decode(next_input_ids, skip_special_tokens=True)
        )
    return sampled_sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to data")
    parser.add_argument("--model_name_or_path", help="Model name or path")
    parser.add_argument(
        "--max_seq_length", type=int, default=64, help="Maximum sequence length"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    parser.add_argument(
        "--num_warmup_samples", type=int, default=5, help="Number of warmup samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bert = get_bert(args.model_name_or_path)
    bert.to(device)

    for target in Target:
        logger.info(f"Assign samples to events with {target} labels")
        dataset = get_dataset(
            args.data_dir, args.model_name_or_path, args.max_seq_length, target
        )
        loader = DataLoader(dataset, args.batch_size)

        logger.info("Start sampling.")
        for _ in tqdm(
            range(args.num_samples + args.num_warmup_samples), desc="# Sampling"
        ):
            dataset.add_sampled_sents(
                sample_sentences(bert, loader, dataset.tokenizer, device)
            )
        dataset.drop_first_n_sampled_sents(args.num_warmup_samples)

        save_event_file(
            dataset.data_dir
            / "{}.{}.{}.jsonl.gz".format(
                target.name.lower(),
                DataType.LABELED.name.lower(),
                Split.TRAIN.name.lower(),
            ),
            dataset.sents,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level="INFO",
    )
    main()
