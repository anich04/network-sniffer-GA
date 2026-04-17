#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  GA Network Capture -- Backend Setup"
echo "============================================================"

# Install deps
echo "[*] Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[*] Starting Flask API on http://localhost:5050"
echo "    Press Ctrl+C to stop."
echo ""
python app.py
