import argparse
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from modules.data_modules import DataModule
from modules.modules import Module
from modules.utils import add_common_argparse_args
from utils.utils import DataType, Split, Target

logger = logging.getLogger(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_argparse_args(parser)
    parser = Module.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    model = Module(args)
    data_module = DataModule.from_argparse_args(args)
    metric_key = "{}_cls_auc".format(
        Module.get_metric_key(
            target=Target[args.val_target],
            data_type=DataType[args.val_data_type],
            split=Split.VAL,
        )
    )
    callbacks = [
        EarlyStopping(monitor=metric_key, mode="max", patience=args.patience),
        ModelCheckpoint(filename="best", monitor=metric_key, mode="max"),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer.from_argparse_args(
        args,
        multiple_trainloader_mode="min_size",
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level="INFO"
    )
    main()
