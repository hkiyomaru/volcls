import argparse
import json
import logging
import pathlib
from pathlib import Path

import pytorch_lightning as pl
import torch

from modules.data_modules import DataModule
from modules.modules import Module
from modules.utils import add_common_argparse_args
from utils.utils import DataType, Split, Target

logger = logging.getLogger(__file__)


class TestScoreLogger(pl.Callback):
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.test_outputs = {}

    def on_test_end(self, trainer, pl_module: Module):
        target = trainer.datamodule.test_target
        data_type = trainer.datamodule.test_data_type
        data_key = pl_module.get_data_key(target, data_type)
        metric_key = pl_module.get_metric_key(target, data_type, Split.TEST)
        self.test_outputs.update(
            {
                data_key: {
                    "auc": pl_module.aucs[metric_key].compute().cpu().item(),
                    "vol_cls_ps": pl_module.test_outputs["vol_cls_ps"],
                    "ani_cls_ps": pl_module.test_outputs["ani_cls_ps"],
                }
            }
        )

    def save_test_outputs(self):
        with (self.log_dir / "test_outputs.json").open("wt") as f:
            json.dump(self.test_outputs, f)

    @staticmethod
    def has_test_log(log_dir: Path):
        return (log_dir / "test_outputs.json").exists()


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_argparse_args(parser)
    parser = Module.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--force", "-f", action="store_true", help="Run test anyway.")
    args = parser.parse_args()

    ckpt_path = Path(args.resume_from_checkpoint)

    log_dir = ckpt_path.parent.parent
    if TestScoreLogger.has_test_log(log_dir) and args.force is False:
        logger.info(f"{log_dir} has been tested already.")
        return

    pl.seed_everything(args.seed)

    model = Module.load_from_checkpoint(ckpt_path)
    test_score_logger = TestScoreLogger(log_dir)
    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=int(torch.cuda.is_available()),
        distributed_backend=None,
        logger=False,
        checkpoint_callback=False,
        callbacks=[test_score_logger],
    )
    for target in Target:
        for data_type in DataType:
            path = (
                pathlib.Path(model.hparams.data_dir)
                / f"{target.name.lower()}.{data_type.name.lower()}.test.jsonl.gz"
            )
            if path.exists():
                data_module = DataModule.from_argparse_args(
                    argparse.Namespace(**model.hparams),
                    test_target=target.name,
                    test_data_type=data_type.name,
                )
                trainer.test(model, datamodule=data_module)
    test_score_logger.save_test_outputs()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level="INFO"
    )
    main()
