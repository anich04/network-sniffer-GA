"""
GA Network Capture — Flask API backend
Endpoints: POST /start  GET /results  GET /fitness  GET /health
"""

import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

from ga_engine     import run_ga
from packet_capture import get_profile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── shared state (protected by a lock) ────────────────────────────────────
_lock    = threading.Lock()
_state   = {
    "status":          "idle",   # idle | running | completed | error
    "mode":            None,
    "results":         None,     # best strategy dict
    "fitness_history": [],       # list[float]
    "error":           None,
    "traffic_profile": None,
}


def _run_in_thread(mode: str):
    """Execute GA in a background thread and update shared state."""
    try:
        profile = get_profile(mode)
        best, history = run_ga(profile)

        with _lock:
            _state["results"]          = best
            _state["fitness_history"]  = history
            _state["traffic_profile"]  = profile
            _state["status"]           = "completed"
            _state["error"]            = None
    except Exception as exc:
        with _lock:
            _state["status"] = "error"
            _state["error"]  = str(exc)


# ── routes ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "offline")

    if mode not in ("offline", "live"):
        return jsonify({"error": "mode must be 'offline' or 'live'"}), 400

    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Optimisation already running"}), 409
        _state["status"]          = "running"
        _state["mode"]            = mode
        _state["results"]         = None
        _state["fitness_history"] = []
        _state["error"]           = None

    t = threading.Thread(target=_run_in_thread, args=(mode,), daemon=True)
    t.start()

    return jsonify({"message": f"GA optimisation started in {mode} mode"}), 202


@app.route("/results", methods=["GET"])
def results():
    with _lock:
        status  = _state["status"]
        res     = _state["results"]
        err     = _state["error"]

    if status == "running":
        return jsonify({"status": "running", "message": "Optimisation in progress"}), 202
    if status == "error":
        return jsonify({"status": "error", "error": err}), 500
    if res is None:
        return jsonify({"status": "idle", "message": "No results yet — run /start first"}), 404

    return jsonify(res)


@app.route("/fitness", methods=["GET"])
def fitness():
    with _lock:
        history = list(_state["fitness_history"])
        status  = _state["status"]

    if not history and status == "running":
        return jsonify({"status": "running", "message": "Still optimising"}), 202
    if not history:
        return jsonify([])

    return jsonify(history)


@app.route("/status", methods=["GET"])
def status():
    """Bonus endpoint — full run metadata for dashboard polling."""
    with _lock:
        return jsonify({
            "status":    _state["status"],
            "mode":      _state["mode"],
            "error":     _state["error"],
            "generations_done": len(_state["fitness_history"]),
        })


if __name__ == "__main__":
    print("=" * 56)
    print("  GA Network Capture — Backend API")
    print("  http://localhost:5050")
    print("  Endpoints: /health  /start  /results  /fitness")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5050, debug=False)
