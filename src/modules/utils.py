import argparse
import enum


class Method(enum.Flag):
    NONE = enum.auto()
    VANILLA = enum.auto()
    WR = enum.auto()
    SOC = enum.auto()
    ADA = enum.auto()


def add_common_argparse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--vol",
        default=Method.VANILLA.name,
        choices=[ty.name for ty in Method],
        help="The method to learn volitionality.",
    )
    parser.add_argument(
        "--ani",
        default=Method.VANILLA.name,
        choices=[ty.name for ty in Method],
        help="The method to learn animacy.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of checks with no improvement after which modules will be stopped",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    return parser
