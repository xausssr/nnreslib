import logging
import os

logging.basicConfig(
    level=logging.INFO if os.environ.get("DEBUG", "0") == "0" else logging.DEBUG,
    format="[%(asctime)s.%(msecs)03d] %(name)-45s %(levelname)8s %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)
