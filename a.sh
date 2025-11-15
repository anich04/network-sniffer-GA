#!/usr/bin/env bash
set -euo pipefail

PROJ="Network-Packet-Analyzer-GA"
VENV_DIR="venv"

echo
echo "=== Network Packet Analyzer (GA) project generator ==="
echo "This will create ./$PROJ with starter code, a venv, and install requirements."
echo

if [ -d "$PROJ" ]; then
  echo "Error: directory '$PROJ' already exists. Please remove or choose another location."
  exit 1
fi

mkdir -p "$PROJ"
cd "$PROJ"

# --------------------------
# FOLDER STRUCTURE
# --------------------------
mkdir -p src/ga_engine src/data/raw src/data/processed notebooks tests/test_pcaps docs

# --------------------------
# .gitignore
# --------------------------
cat > .gitignore <<'GIT'
__pycache__/
*.pyc
*.pkl
*.sqlite3
venv/
env/
.env
src/data/raw/*.pcap
src/data/*.csv
.DS_Store
GIT

# --------------------------
# requirements.txt
# --------------------------
cat > requirements.txt <<'REQ'
scapy
pandas
numpy
matplotlib
deap
scikit-learn
joblib
REQ

# --------------------------
# README.md
# --------------------------
cat > README.md <<'MD'
# Network Packet Analyzer (Genetic Algorithm)

This project is a simplified Python-only system for capturing packets, extracting features,
training a Genetic Algorithm (DEAP) model, and analyzing anomalies.

Follow the setup script instructions printed during generation.
MD

# Empty placeholder pcap
touch src/data/raw/sample.pcap


# --------------------------
# packet_capture.py
# --------------------------
cat > src/packet_capture.py <<'PY'
#!/usr/bin/env python3
import argparse
from scapy.all import rdpcap, sniff, wrpcap

def read_pcap(path):
    print(f"Reading pcap: {path}")
    pkts = rdpcap(path)
    print(f"Loaded {len(pkts)} packets")
    return pkts

def sniff_live(count=10, iface=None, out_path="src/data/raw/live_capture.pcap"):
    print(f"Sniffing {count} packets...")
    pkts = sniff(count=count, iface=iface)
    wrpcap(out_path, pkts)
    print(f"Saved live capture to {out_path}")
    return pkts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", help="Path to .pcap file")
    parser.add_argument("--sniff", action="store_true")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    if args.pcap:
        read_pcap(args.pcap)
    elif args.sniff:
        sniff_live(count=args.count)
    else:
        print("Use --pcap <file> or --sniff")

if __name__ == "__main__":
    main()
PY


# --------------------------
# extract_features.py
# --------------------------
cat > src/extract_features.py <<'PY'
#!/usr/bin/env python3
import argparse
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP

def pkt_to_features(pkt):
    row = {
        "src_ip": None, "dst_ip": None,
        "src_port": None, "dst_port": None,
        "protocol": None, "size": len(pkt),
        "ttl": None, "flags": None,
        "timestamp": float(getattr(pkt, "time", 0.0))
    }
    if IP in pkt:
        row["src_ip"] = pkt[IP].src
        row["dst_ip"] = pkt[IP].dst
        row["ttl"] = pkt[IP].ttl
        row["protocol"] = pkt[IP].proto
    if TCP in pkt:
        row["src_port"] = pkt[TCP].sport
        row["dst_port"] = pkt[TCP].dport
        row["flags"] = str(pkt[TCP].flags)
        row["protocol"] = "TCP"
    elif UDP in pkt:
        row["src_port"] = pkt[UDP].sport
        row["dst_port"] = pkt[UDP].dport
        row["protocol"] = "UDP"
    return row

def extract(pcap, out_csv):
    pkts = rdpcap(pcap)
    rows = [pkt_to_features(p) for p in pkts]
    df = pd.DataFrame(rows)
    df["time_diff"] = df["timestamp"].diff().fillna(0)
    df.to_csv(out_csv, index=False)
    print(f"Saved features to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", default="src/data/raw/sample.pcap")
    parser.add_argument("--out", default="src/data/processed/features.csv")
    args = parser.parse_args()
    extract(args.pcap, args.out)

if __name__ == "__main__":
    main()
PY


# --------------------------
# utils.py
# --------------------------
cat > src/utils.py <<'PY'
import json
import os

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)
PY


# --------------------------
# analyzer.py
# --------------------------
cat > src/analyzer.py <<'PY'
#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib

def apply_ga(df, model):
    thr = model.get("thresholds", {})
    size_t = thr.get("size_norm", 3.0)
    df["suspicious"] = df.get("size_norm", 0) > size_t
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="src/data/processed/features.csv")
    parser.add_argument("--model", default="src/ga_engine/ga_model.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    model = joblib.load(args.model)
    df = apply_ga(df, model)
    df.to_csv("src/data/processed/results.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()
PY


# --------------------------
# GA fitness function
# --------------------------
cat > src/ga_engine/fitness_function.py <<'PY'
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def evaluate_feature_subset(X, y, selected):
    if sum(selected) == 0:
        return 0.0
    cols = [i for i,b in enumerate(selected) if b==1]
    Xs = X[:, cols]
    clf = RandomForestClassifier(n_estimators=40)
    scores = cross_val_score(clf, Xs, y, cv=3, scoring="f1_macro")
    return scores.mean()
PY


# --------------------------
# ga_train.py
# --------------------------
cat > src/ga_engine/ga_train.py <<'PY'
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import random
import joblib
from deap import base, creator, tools, algorithms
from fitness_function import evaluate_feature_subset

DATA = "src/data/processed/features.csv"
OUT = "src/ga_engine/ga_model.pkl"

def main():
    df = pd.read_csv(DATA)
    cols = [c for c in df.columns if df[c].dtype != object and c not in ("label",)]
    X = df[cols].fillna(0).values
    if "label" in df:
        y = df["label"].values
    else:
        y = (df["size"] > df["size"].quantile(0.98)).astype(int).values

    n = X.shape[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def eval_ind(ind):
        return (evaluate_feature_subset(X, y, ind),)

    toolbox.register("evaluate", eval_ind)

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=15,
                                   stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    model = {"selected": list(best), "score": best.fitness.values[0]}
    joblib.dump(model, OUT)
    print("Saved GA model to", OUT)

if __name__ == "__main__":
    main()
PY


# --------------------------
# apply_ga_rules.py
# --------------------------
cat > src/ga_engine/apply_ga_rules.py <<'PY'
#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np
import json

MODEL="src/ga_engine/ga_model.pkl"
DATA="src/data/processed/features.csv"
OUT="src/ga_engine/ga_rules.json"

def main():
    model = joblib.load(MODEL)
    df = pd.read_csv(DATA)

    # simple heuristic threshold builder
    rules = {"thresholds": {}}
    if "size_norm" in df.columns:
        rules["thresholds"]["size_norm"] = float(df["size_norm"].quantile(0.98))

    with open(OUT, "w") as f:
        json.dump(rules, f, indent=2)

    print("Wrote GA rules to", OUT)

if __name__ == "__main__":
    main()
PY


# --------------------------
# Virtualenv + pip install
# --------------------------
echo
echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"

PIP="$PWD/$VENV_DIR/bin/pip"
PYBIN="$PWD/$VENV_DIR/bin/python"

if [ -x "$PIP" ]; then
  "$PIP" install --upgrade pip setuptools wheel
  "$PIP" install -r requirements.txt
else
  echo "Warning: pip not found in venv. Install manually."
fi


# --------------------------
# Now safe to create placeholder GA model
# --------------------------
echo "Creating placeholder GA model (now joblib is installed)..."

$PYBIN - <<'PYCODE'
import joblib, os
model = {"thresholds":{"size_norm":3.0}, "selected":[], "score":0.0}
os.makedirs("src/ga_engine", exist_ok=True)
joblib.dump(model, "src/ga_engine/ga_model.pkl")
print("Placeholder GA model created.")
PYCODE


# --------------------------
# Done
# --------------------------
echo
echo "=== Setup complete ==="
echo "Project created: $(pwd)"
echo
echo "Next steps:"
echo "1) Activate venv:"
echo "   source venv/bin/activate"
echo
echo "2) Extract features:"
echo "   python src/extract_features.py --pcap src/data/raw/sample.pcap"
echo
echo "3) Train GA:"
echo "   python src/ga_engine/ga_train.py"
echo
echo "4) Apply GA rules:"
echo "   python src/ga_engine/apply_ga_rules.py"
echo
echo "5) Analyze traffic:"
echo "   python src/analyzer.py"
echo
