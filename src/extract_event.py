import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from labeling.extractor import EnglishExtractor, JapaneseExtractor
from labeling.labeler import EnglishLabeler, JapaneseLabeler
from utils.utils import DataType, Event, Target, save_event_file

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True, help="Path to data file")
    parser.add_argument("--output_dir", required=True, help="Path to output file")
    parser.add_argument(
        "--vol_seed_lex_file", required=True, help="Path to the seed lexicon"
    )
    parser.add_argument(
        "--ani_seed_lex_file",
        required=False,
        default=None,
        help="Path to the seed lexicon of animate/inanimate words",
    )
    parser.add_argument(
        "--lang", required=True, choices=("ja", "en"), help="Language indicator"
    )
    args = parser.parse_args()

    if args.lang == "ja":
        extractor = JapaneseExtractor(args.data_file, args.vol_seed_lex_file)
        # As for animacy, use the dictionary integrated into KNP
        labeler = JapaneseLabeler(args.data_file, args.vol_seed_lex_file, None)
    elif args.lang == "en":
        extractor = EnglishExtractor(args.data_file, args.vol_seed_lex_file)
        labeler = EnglishLabeler(
            args.data_file, args.vol_seed_lex_file, args.ani_seed_lex_file
        )
    else:
        raise ValueError

    events: Dict[Tuple[Target, DataType], List[Event]] = {}
    for target in Target:
        for data_type in DataType:
            events[(target, data_type)] = []

    for event in map(labeler, extractor(filter_by_seed=True)):
        if event.vol_label != -100:
            events[(Target.VOL, DataType.LABELED)].append(event)

    n = len(events[(Target.VOL, DataType.LABELED)])

    logger.info(f"# of ({Target.VOL.name}/{DataType.LABELED.name}): {n}")
    if n == 0:
        return
    for event in map(labeler, extractor()):
        if event.vol_label == -100:
            events[(Target.VOL, DataType.UNLABELED)].append(event)
        if event.ani_label != -100:
            events[(Target.ANI, DataType.LABELED)].append(event)
        if event.ani_label == -100:
            events[(Target.ANI, DataType.UNLABELED)].append(event)
        if all(len(v) >= n for v in events.values()):
            break
    m = n
    for target in Target:
        for data_type in (DataType.LABELED, DataType.UNLABELED):
            m = min(m, len(events[(target, data_type)]))
    if m == 0:
        return

    events = {k: v[:m] for k, v in events.items()}

    output_dir = Path(args.output_dir)
    stem = Path(Path(args.data_file).name)
    while str(stem) != stem.stem:
        stem = Path(stem.stem)
    for target in Target:
        for data_type in (DataType.LABELED, DataType.UNLABELED):
            filename = (
                f"{str(stem)}.{target.name.lower()}.{data_type.name.lower()}.jsonl.gz"
            )
            save_event_file(output_dir / filename, events[(target, data_type)])


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level="INFO",
    )
    main()
