import logging

logger = logging.getLogger("CryptoLab")

logger.setLevel(logging.INFO)

console = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console.setFormatter(formatter)

logger.addHandler(console)