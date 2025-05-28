# shared.py
from threading import Lock

latest_prediction_data = {
    "price": 0,
    "signal": None,
    "confidence": 0,
    "timestamp": None,
    "supply_zones": [],
    "demand_zones": [],
}

data_lock = Lock()
