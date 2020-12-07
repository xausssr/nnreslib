from argparse import ArgumentParser, Namespace

from . import __version__


def parse_args() -> Namespace:
    parser = ArgumentParser(description="nnreslib")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--msg", required=True, help="Specify output message")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Hello!", args.msg)


if __name__ == "__main__":
    main()
