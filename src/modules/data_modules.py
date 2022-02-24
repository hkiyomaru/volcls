import argparse
from pathlib import Path
from typing import Dict, Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from modules.dataset import EvaluationDataset, TrainingDataset
from modules.utils import Method
from utils.utils import DataType, Split, Target


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        vol: str = Method.VANILLA.name,
        ani: str = Method.VANILLA.name,
        num_workers: int = 0,
        max_seq_length: int = 64,
        model_name_or_path: str = "",
        batch_size: int = 64,
        val_target: str = Target.VOL.name,
        val_data_type: str = DataType.UNLABELED.name,
        test_target: str = Target.VOL.name,
        test_data_type: str = DataType.UNLABELED.name,
        seed_lex: Optional[str] = None,
        limit_train_data: Optional[int] = None,
    ):
        super(DataModule, self).__init__()
        self.data_dir = Path(data_dir)
        self.methods = {Target.VOL: Method[vol], Target.ANI: Method[ani]}
        assert not (self.methods[Target.VOL] == self.methods[Target.ANI] == Method.NONE)
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size

        self.val_target = Target[val_target]
        self.val_data_type = DataType[val_data_type]
        self.test_target = Target[test_target]
        self.test_data_type = DataType[test_data_type]

        self.seed_lex = seed_lex
        self.limit_train_data = limit_train_data

        self.train_datasets: Dict[Target, Dataset] = {}
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_datasets = {
                Target.VOL: TrainingDataset(
                    data_dir=self.data_dir,
                    model_name_or_path=self.model_name_or_path,
                    max_seq_length=self.max_seq_length,
                    seed_lex=self.seed_lex,
                    limit_train_data=self.limit_train_data,
                    target=Target.VOL,
                ),
                Target.ANI: TrainingDataset(
                    data_dir=self.data_dir,
                    model_name_or_path=self.model_name_or_path,
                    max_seq_length=self.max_seq_length,
                    target=Target.ANI,
                ),
            }
            self.val_dataset = EvaluationDataset(
                data_dir=self.data_dir,
                model_name_or_path=self.model_name_or_path,
                max_seq_length=self.max_seq_length,
                target=self.val_target,
                data_type=self.val_data_type,
                split=Split.VAL,
            )
        if stage == "test" or stage is None:
            self.test_dataset = EvaluationDataset(
                data_dir=self.data_dir,
                model_name_or_path=self.model_name_or_path,
                max_seq_length=self.max_seq_length,
                target=self.test_target,
                data_type=self.test_data_type,
                split=Split.TEST,
            )

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        # Use CombinedLoader to inform the target and data type
        return CombinedLoader(
            {
                (self.val_target, self.val_data_type): DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            }
        )

    def test_dataloader(self):
        # Use CombinedLoader to inform the target and data type
        return CombinedLoader(
            {
                (self.test_target, self.test_data_type): DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            }
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser, **kwargs):
        parser.add_argument("--data_dir", help="Data directory")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        parser.add_argument(
            "--max_seq_length", type=int, default=48, help="Maximum sequence length"
        )
        parser.add_argument(
            "--num_workers", type=int, default=0, help="Parallelism of data loading"
        )
        parser.add_argument(
            "--val_target",
            default=Target.VOL.name,
            choices=[ty.name for ty in Target],
            help="The target of validation dataset.",
        )
        parser.add_argument(
            "--val_data_type",
            default=DataType.UNLABELED.name,
            choices=[ty.name for ty in DataType],
            help="The data type of validation dataset.",
        )
        parser.add_argument(
            "--test_target",
            default=Target.VOL.name,
            choices=[ty.name for ty in Target],
            help="The target of test dataset.",
        )
        parser.add_argument(
            "--test_data_type",
            default=DataType.UNLABELED.name,
            choices=[ty.name for ty in DataType],
            help="The data type of test dataset.",
        )
        parser.add_argument(
            "--num_soc_samples",
            default=3,
            help="The number of samples used for SOC.",
        )
        parser.add_argument(
            "--seed_lex",
            default=None,
            help="Path to the seed lexicon that lists volitional/non-volitional adverbs used for training.",
        )
        parser.add_argument(
            "--limit_train_data",
            type=int,
            default=None,
            help="If set, restrict the number of training data.",
        )
        return parser
