from argparse import ArgumentParser, Namespace

from . import __version__
from .func import my_func


def parse_args() -> Namespace:
    parser = ArgumentParser(description="nnreslib")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--msg", required=True, help="Specify output message")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = my_func(10)
    print("Hello!", args.msg, result)


if __name__ == "__main__":
    main()
