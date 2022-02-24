import argparse
import collections
import json
import logging
import pathlib
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup

from modules.functions import reverse_grad
from modules.models import Classifier, Discriminator, Encoder
from modules.utils import Method
from utils.utils import DataType, Split, Target

logger = logging.getLogger(__file__)


class Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if not isinstance(hparams, argparse.Namespace):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)

        self.methods = {
            Target.VOL: Method[hparams.vol],
            Target.ANI: Method[hparams.ani],
        }
        assert not (self.methods[Target.VOL] == self.methods[Target.ANI] == Method.NONE)

        self.alpha: Dict[Target, float] = {
            Target.VOL: hparams.alpha_vol,
            Target.ANI: hparams.alpha_ani,
        }
        self.beta = hparams.beta

        self.cfg = BertConfig.from_pretrained(hparams.model_name_or_path)

        self.tokenizer = BertTokenizer.from_pretrained(hparams.model_name_or_path)

        # Common modules.
        self.enc = Encoder.from_pretrained(hparams.model_name_or_path)
        self.clss = nn.ModuleDict(
            {
                Target.VOL.name: Classifier(self.cfg.hidden_size),
                Target.ANI.name: Classifier(self.cfg.hidden_size),
            }
        )

        # SOC
        self.num_soc_samples = hparams.num_soc_samples

        # Modules for ADA.
        diss = {}
        if self.methods[Target.VOL] == Method.ADA:
            diss[Target.VOL.name] = Discriminator(self.cfg.hidden_size)
        if self.methods[Target.ANI] == Method.ADA:
            diss[Target.ANI.name] = Discriminator(self.cfg.hidden_size)
        self.diss = nn.ModuleDict(diss)

        # ROC-AUC.
        aucs = {}
        for target in Target:
            for data_type in DataType:
                for split in Split:
                    metric_key = self.get_metric_key(target, data_type, split)
                    aucs[metric_key] = torchmetrics.AUROC(
                        pos_label=1, compute_on_step=split == Split.TRAIN
                    )
        self.aucs = nn.ModuleDict(aucs)

        # Validation metric information.
        self.val_target = Target[hparams.val_target]
        self.val_data_type = DataType[hparams.val_data_type]

        # Misc.
        self.test_outputs = None

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        pooler_output: torch.Tensor = None,
        labels: torch.Tensor = None,
        target: Target = Target.VOL,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pooler_output is None:
            pooler_output = self.enc(input_ids, attention_mask, token_type_ids)
        cls_logits, cls_loss = self.clss[target.name](pooler_output, labels)
        return cls_logits, cls_loss, pooler_output

    def training_step(
        self,
        batches: Dict[Tuple[Target, Method], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        total_loss = 0
        for target, batch in batches.items():
            total_loss += self._training_step(batch, target)
        return total_loss

    def _training_step(self, batch: Dict[str, torch.Tensor], target: Target):
        metric_key = self.get_metric_key(target, DataType.LABELED, Split.TRAIN)

        loss = 0

        # Learn consistency on unlabeled data.
        if target == Target.VOL and self.beta > 0.0:
            vol_cls_logits, _, pooler_output_unlabeled = self(
                input_ids=batch["input_ids_unlabeled"],
                attention_mask=batch["attention_mask_unlabeled"],
                token_type_ids=batch["token_type_ids_unlabeled"],
                target=Target.VOL,
            )
            ani_cls_logits, _, _ = self(
                pooler_output=pooler_output_unlabeled, target=Target.ANI
            )
            con_loss = torch.mean(
                torch.maximum(
                    vol_cls_logits - ani_cls_logits, torch.zeros_like(vol_cls_logits)
                )
            )
            loss += self.beta * con_loss
            self.log(f"{metric_key}_con_loss", con_loss)

        if self.methods[target] == Method.NONE:
            return loss

        # Learn classification using labeled data.
        cls_logits, cls_loss, pooler_output = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            target=target,
        )
        loss += cls_loss
        self.log(f"{metric_key}_cls_loss", cls_loss)
        if torch.max(batch["labels"] - torch.min(batch["labels"])).item() != 0:
            # AUC cannot be calculated with a batch consisting only positive or negative data.
            self.log(
                f"{metric_key}_cls_auc",
                self.aucs[metric_key](
                    torch.sigmoid(cls_logits).view(-1), batch["labels"].view(-1).int()
                ),
            )

        # Learn by WR.
        if self.methods[target] == Method.WR:
            _, cls_loss_wr, _ = self(
                input_ids=batch["input_ids_wr"],
                attention_mask=batch["attention_mask_wr"],
                token_type_ids=batch["token_type_ids_wr"],
                labels=batch["labels"],
                target=target,
            )
            loss += self.alpha[target] * cls_loss_wr
            self.log(f"{metric_key}_wr_loss", cls_loss_wr)

        # Learn by SOC.
        if self.methods[target] == Method.SOC:
            soc_loss = 0
            for i in range(self.num_soc_samples):
                cls_logits_soc_s, _, _ = self(
                    input_ids=batch[f"input_ids_soc_s_{i}"],
                    attention_mask=batch[f"attention_mask_soc_s_{i}"],
                    token_type_ids=batch[f"token_type_ids_soc_s_{i}"],
                    target=target,
                )
                cls_logits_soc_soc, _, _ = self(
                    input_ids=batch[f"input_ids_soc_soc_{i}"],
                    attention_mask=batch[f"attention_mask_soc_soc_{i}"],
                    token_type_ids=batch[f"token_type_ids_soc_soc_{i}"],
                    target=target,
                )
                soc_loss += torch.mean((cls_logits_soc_s - cls_logits_soc_soc) ** 2)
            loss += self.alpha[target] * (soc_loss / self.num_soc_samples)
            self.log(f"{metric_key}_soc_loss", soc_loss)

        # Learn by ADA.
        if self.methods[target] == Method.ADA:
            _, _, pooler_output_unlabeled = self(
                input_ids=batch["input_ids_unlabeled"],
                attention_mask=batch["attention_mask_unlabeled"],
                token_type_ids=batch["token_type_ids_unlabeled"],
                target=Target.VOL,
            )
            _, ada_loss = self.diss[target.name](
                pooler_output=reverse_grad(
                    torch.cat([pooler_output, pooler_output_unlabeled]),
                    alpha=self.alpha[target],
                ),
                labels=torch.cat(
                    [
                        torch.ones_like(batch["labels"]),
                        torch.zeros_like(batch["labels"]),
                    ]
                ),
            )
            loss += ada_loss
            self.log(f"{metric_key}_ada_loss", ada_loss)

        return loss

    def validation_step(
        self,
        batches: Dict[Tuple[Target, DataType], Dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        for (target, data_type), batch in batches.items():
            metric_key = self.get_metric_key(target, data_type, Split.VAL)
            cls_logits, cls_loss, _ = self(**batch, target=target)
            self.log(f"{metric_key}_cls_loss", cls_loss)
            self.aucs[metric_key](
                torch.sigmoid(cls_logits).view(-1), batch["labels"].view(-1).int()
            )
            self.log(
                f"{metric_key}_cls_auc",
                self.aucs[metric_key],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def test_step(
        self,
        batches: Dict[Tuple[Target, DataType], Dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        for (target, data_type), batch in batches.items():
            metric_key = self.get_metric_key(target, data_type, Split.TEST)
            vol_cls_logits, _, pooler_output = self(**batch, target=Target.VOL)
            ani_cls_logits, _, _ = self(pooler_output=pooler_output, target=Target.ANI)
            if target == Target.VOL:
                cls_logits = vol_cls_logits
            else:
                cls_logits = ani_cls_logits
            self.aucs[metric_key](
                torch.sigmoid(cls_logits).view(-1), batch["labels"].view(-1).int()
            )
            self.log(
                f"{metric_key}_cls_auc",
                self.aucs[metric_key],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return {
                "vol_cls_ps": torch.sigmoid(vol_cls_logits),
                "ani_cls_ps": torch.sigmoid(ani_cls_logits),
            }

    def test_epoch_end(self, outputs):
        self.test_outputs = collections.defaultdict(list)
        for output in outputs:
            for key, value in output.items():
                self.test_outputs[key].extend(value.cpu().tolist())

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        vol_cls_logits, _, pooler_output = self(**batch, target=Target.VOL)
        ani_cls_logits, _, _ = self(pooler_output=pooler_output, target=Target.ANI)
        return {
            "vol_cls_ps": torch.sigmoid(vol_cls_logits),
            "ani_cls_ps": torch.sigmoid(ani_cls_logits),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_training_steps * self.hparams.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @property
    def num_training_steps(self) -> int:
        """Total modules steps inferred from datamodule and devices."""
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = min(
                len(dl) for dl in self.trainer.datamodule.train_dataloader().values()
            )
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            raise ValueError

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)

        batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, _) -> None:
        data_key = self.get_data_key(self.val_target, self.val_data_type)
        metric_key = self.get_metric_key(self.val_target, self.val_data_type, Split.VAL)
        auc = self.aucs[metric_key].compute().cpu().item()
        with (pathlib.Path(self.logger.log_dir) / "val_outputs.json").open("wt") as f:
            json.dump({data_key: {"auc": auc}}, f)

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--model_name_or_path", help="Path to BERT model.")
        parser.add_argument(
            "--learning_rate", type=float, default=5e-5, help="Learning rate."
        )
        parser.add_argument(
            "--alpha_vol",
            type=float,
            default=1.0,
            help="Prevent the model from relying on the seed words in volitionality classification.",
        )
        parser.add_argument(
            "--alpha_ani",
            type=float,
            default=1.0,
            help="Prevent the model from relying on the seed words in animacy classification.",
        )
        parser.add_argument(
            "--beta",
            type=float,
            default=1.0,
            help="Prevent the model from producing inconsistent predictions.",
        )
        parser.add_argument(
            "--num_warmup_steps",
            type=float,
            default=0.1,
            help="Percentage of modules steps to use as warmup.",
        )
        return parser

    @staticmethod
    def get_data_key(target: Target, data_type: DataType) -> str:
        return f"{target.name.lower()}_{data_type.name.lower()}"

    @staticmethod
    def get_metric_key(target: Target, data_type: DataType, split: Split) -> str:
        return f"{target.name.lower()}_{data_type.name.lower()}_{split.name.lower()}"
