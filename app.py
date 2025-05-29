from flask import Flask, jsonify
from shared import latest_prediction_data, data_lock
import threading
from vertix_trader import run_predictor

app = Flask(__name__)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/prediction")
def prediction():
    with data_lock:
        return latest_prediction_data

def start_background_worker():
    thread = threading.Thread(target=run_predictor, daemon=True)
    thread.start()

if __name__ == "__main__":
    start_background_worker()
    app.run(host="0.0.0.0", port=5000)
