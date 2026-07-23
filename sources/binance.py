"""
CryptoLab DataHub
Binance Downloader
Release 0.1.0
"""

from __future__ import annotations

import requests
import pandas as pd

from config import (
    BASE_URL,
    REQUEST_TIMEOUT,
    LIMIT,
)

from utils.logger import logger


class BinanceClient:

    def __init__(self):

        logger.info("Initializing Binance Client...")

        self.base_url = BASE_URL
        self.timeout = REQUEST_TIMEOUT
        self.limit = LIMIT

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": "CryptoLab-DataHub/0.1.0"
        })

    def download(
        self,
        symbol: str,
        interval: str,
        start_time: int
    ):

        logger.info(
            f"Downloading {symbol} {interval}"
        )

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": self.limit,
        }

        response = self.session.get(
            self.base_url,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        logger.info(
            f"Downloaded {len(data)} candles."
        )

        columns = [
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Volume",
            "Trades",
            "Taker Base",
            "Taker Quote",
            "Ignore",
        ]

        df = pd.DataFrame(
            data,
            columns=columns,
        )

        return df