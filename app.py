# from flask import Flask, jsonify
# from flask_cors import CORS
# from threading import Lock

# app = Flask(__name__)
# CORS(app)

# latest_prediction_data = {
#     "price": None,
#     "signal": None,
#     "confidence": None,
#     "timestamp": None,
#     "supply_zones": [],
#     "demand_zones": []
# }
# data_lock = Lock()

# @app.route("/predict", methods=["GET"])
# def get_prediction():
#     with data_lock:
#         if latest_prediction_data["price"] is None:
#             return jsonify({"error": "Prediction not available yet"}), 503
#         return jsonify(latest_prediction_data)

# @app.route("/health")
# def health():
#     return {"status": "ok"}, 200




# def start_api():
#     app.run(host="0.0.0.0", port=5000, debug=False)


# if __name__ == "__main__":
#     start_api()




from flask import Flask, jsonify
import threading
from vertix-trader import run_predictor  # Import your loop

app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify(status="ok")

# Start predictor in background
def start_background_worker():
    thread = threading.Thread(target=run_predictor, daemon=True)
    thread.start()

if __name__ == "__main__":
    start_background_worker()
    app.run(host="0.0.0.0", port=5000)
