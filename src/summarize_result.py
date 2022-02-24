import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from modules.utils import Method
from utils.utils import DataType, Target

logger = logging.getLogger(__file__)

ExpId = Tuple[str, str, bool, bool, bool]
DataId = Tuple[Target, DataType]


def parse_tag(tag: str) -> Tuple[Method, Method, float, float, float]:
    vol, ani, alpha_vol, alpha_ani, beta = tag.split("_")
    return Method[vol], Method[ani], float(alpha_vol), float(alpha_ani), float(beta)


def parse_result(paths: List[Path]) -> Tuple[Dict[DataId, float], Dict[DataId, float]]:
    aucs: Dict[Tuple[Target, DataType], List[float]] = collections.defaultdict(list)
    for path in paths:
        with path.open("rt") as f:
            outputs = json.load(f)
            for target in Target:
                for data_type in DataType:
                    if data_type == DataType.ALL:
                        continue
                    result = outputs.get(
                        f"{target.name.lower()}_{data_type.name.lower()}", None
                    )
                    if result:
                        aucs[(target, data_type)].append(result["auc"])
    return (
        {key: np.mean(scores) for key, scores in aucs.items()},
        {key: np.std(scores) for key, scores in aucs.items()},
    )


def show_summary(args: argparse.Namespace):
    result_dir = Path(args.result_dir)
    val_results = {}  # type: Dict[ExpId, float]
    test_results = collections.defaultdict(
        dict
    )  # type: Dict[ExpId, Dict[str, Dict[str, float]]]
    for log_path in result_dir.glob("**/lightning_logs"):
        # Find test outputs.
        val_outputs_paths = list(log_path.glob("**/val_outputs.json"))
        test_outputs_paths = list(log_path.glob("**/test_outputs.json"))
        if len(test_outputs_paths) == 0:
            continue

        # Parse the experiment setting.
        vol, ani, alpha_vol, alpha_ani, beta = parse_tag(log_path.parent.name)
        alpha_vol = alpha_vol > 0
        alpha_ani = alpha_ani > 0
        beta = beta > 0
        exp_id: ExpId = (vol.value, ani.value, alpha_vol, alpha_ani, beta)

        # Extract the AUC for each type of the test sets.
        val_auc_avgs, _ = parse_result(val_outputs_paths)
        test_auc_avgs, test_auc_stds = parse_result(test_outputs_paths)

        # Compare the score to one with different hyper parameters.
        val_auc_avg = val_auc_avgs[(Target.VOL, DataType.UNLABELED)]
        if exp_id in val_results:
            best_val_auc_avg = val_results[exp_id]
            if best_val_auc_avg > val_auc_avg:
                continue

        # Update the results.
        val_results[exp_id] = val_auc_avg
        for target, data_type in test_auc_avgs.keys():
            test_results[exp_id][target.name + "/" + data_type.name] = {
                "avg": test_auc_avgs[(target, data_type)],
                "std": test_auc_stds[(target, data_type)],
            }

    results = collections.defaultdict(list)
    for (vol, ani, alpha_vol, alpha_ani, beta), scores in test_results.items():
        results["VOL"].append(vol)
        results["ANI"].append(ani)
        results["CO"].append("+ CO" if beta else "")
        for ty, auc in scores.items():
            results[ty].append(f"{auc['avg'] * 100:.1f}±{auc['std'] * 100:.1f}")
    results = pd.DataFrame.from_dict(results)
    results.set_index(
        [Target.VOL.name, Target.ANI.name, "CO"],
        inplace=True,
    )
    results.sort_index(inplace=True)
    results.index = results.index.map(
        lambda index: (
            Method(index[0]).name,
            Method(index[1]).name,
            index[2],
        )
    )
    print(results.to_latex().replace("±", "$\\pm$"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True, help="Result directory.")
    args = parser.parse_args()

    show_summary(args)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level="INFO"
    )
    main()
