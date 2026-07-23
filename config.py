"""
CryptoLab DataHub
Global Configuration
Release: 0.1.0
"""

# Binance API
BASE_URL = "https://api.binance.com/api/v3/klines"

# Network
REQUEST_TIMEOUT = 30
LIMIT = 1000

# Data
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"

# Output
DATA_DIR = "data"
LOG_DIR = "logs"
# Binance
KLINES_ENDPOINT = "/api/v3/klines"

# Retry
MAX_RETRIES = 3
RETRY_DELAY = 2