from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from . import __version__

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s.%(msecs)03d] %(name)-20s %(levelname)8s %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="nnreslib")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--msg", required=True, help="Specify output message")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Hello! %s", args.msg)


if __name__ == "__main__":
    main()
