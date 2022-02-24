import argparse
import logging
from pathlib import Path

import tqdm

from utils.utils import DataType, Split, Target, read_event_file, save_event_file

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to data.")
    parser.add_argument("--output_dir", help="Path to output.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    for target in Target:
        for data_type in (DataType.LABELED, DataType.UNLABELED):
            logger.info(f"Read sentences of ({target}, {data_type}).")
            glob_pat = f"**/*.{target.name.lower()}.{data_type.name.lower()}.jsonl.gz"
            sents = []
            for path in tqdm.tqdm(
                data_dir.glob(glob_pat), desc=f"{target}/{data_type}"
            ):
                sents.extend(read_event_file(path))
            logger.info(f"Done. Found {len(sents)} sentences.")

            logger.info(f"Save sentences of ({target}, {data_type}).")
            filename = f"{target.name.lower()}.{data_type.name.lower()}.{Split.TRAIN.name.lower()}.jsonl.gz"
            save_event_file(output_dir / filename, sents)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level="INFO",
    )
    main()
