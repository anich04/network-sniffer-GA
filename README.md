# GA Network Capture — Genetic Algorithm Based Network Traffic Capture System

A full-stack cybersecurity analytics tool that uses a **Genetic Algorithm** to evolve and optimise the best network packet capture strategy, then visualises the results in a professional dark-theme dashboard.

---

## Overview

Traditional network capture rules are manually tuned and static. This project evolves capture strategies automatically using a Genetic Algorithm:

- **Chromosome** — `(protocol, port_min, port_max, sampling_rate)`
- **Fitness** — scored on protocol match, hot-port coverage, range efficiency, sampling rate, port entropy, and flow throughput
- **Evolution** — 60-individual population, 40 generations, tournament selection, uniform crossover, elitism, and per-gene mutation

---

## Project Structure

```
├── index.html              # Frontend dashboard (single-file, no build step)
└── backend/
    ├── app.py              # Flask REST API
    ├── ga_engine.py        # Genetic Algorithm core
    ├── packet_capture.py   # Traffic profiling (offline + live via psutil)
    ├── requirements.txt
    ├── start.bat           # Windows launcher
    └── start.sh            # Linux / macOS launcher
```

---

## Quick Start

### 1. Start the Backend

**Windows**
```bat
cd backend
start.bat
```

**Linux / macOS**
```bash
cd backend
chmod +x start.sh && ./start.sh
```

The API starts on **`http://localhost:5050`**.

### 2. Expose via ngrok (optional — for remote / demo access)

```bash
ngrok http 5050
```

Copy the `https://…ngrok-free.app` URL.

### 3. Open the Dashboard

Open `index.html` in any browser, paste the backend URL into the URL bar, and click **Check Health** to verify connectivity.

---

## API Reference

| Method | Endpoint  | Description |
|--------|-----------|-------------|
| GET    | `/health` | Returns `{"status": "ok"}` |
| POST   | `/start`  | Start GA optimisation — body: `{"mode": "offline" \| "live"}` |
| GET    | `/results`| Best capture strategy after completion |
| GET    | `/fitness`| Array of best-fitness values per generation |
| GET    | `/status` | Run metadata (status, mode, generations done) |

### Example — start optimisation

```bash
curl -X POST http://localhost:5050/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "offline"}'
```

### Example — fetch best strategy

```bash
curl http://localhost:5050/results
```

```json
{
  "protocol": "TCP",
  "port_min": 442,
  "port_max": 8443,
  "sampling_rate": 0.75
}
```

### Example — fetch fitness history

```bash
curl http://localhost:5050/fitness
```

```json
[8.12, 10.34, 12.87, 14.50, 15.91, 16.44, 17.02, 17.38]
```

---

## Modes

| Mode    | How traffic profile is built |
|---------|------------------------------|
| Offline | Synthetic profile using realistic port-frequency distributions |
| Live    | Reads real NIC counters via `psutil` for 2 s, derives actual traffic characteristics |

---

## Genetic Algorithm Details

| Parameter       | Value |
|-----------------|-------|
| Population size | 60    |
| Generations     | 40    |
| Selection       | Tournament (k = 5) |
| Crossover prob  | 0.80 (single-point) |
| Mutation prob   | 0.25 per individual, 0.40 per gene |
| Elitism         | Top 4 carried forward unchanged |

### Fitness Components

| Component          | Weight | Description |
|--------------------|--------|-------------|
| Protocol match     | 3.0    | Matches dominant protocol in traffic profile |
| Hot-port coverage  | 4.0    | Fraction of high-traffic ports inside the capture range |
| Range efficiency   | 2.0    | Penalises overly narrow or overly broad ranges |
| Sampling rate      | 2.5    | Log-scaled reward for higher sampling |
| Port entropy       | 1.5    | Diversity bonus for wider port ranges |
| Flow throughput    | 2.0    | Estimated packets captured relative to total flows |

---

## Dashboard Features

- Dark glassmorphism UI — deep navy + neon blue / purple accents
- Real-time status badge (Idle / Running / Completed / Error)
- Live generation counter while GA is running
- Animated metric cards — Packet Count, Flow Count, Port Entropy, Protocol Diversity, Fitness Score
- Chart.js fitness evolution curve with gradient fill
- Clean error handling with retry button
- Backend URL saved to `localStorage`

---

## Dependencies

**Backend**
```
flask>=3.0.0
flask-cors>=4.0.0
psutil>=5.9.0
```

**Frontend** — loaded from CDN, no install required
- [Chart.js 4.4](https://www.chartjs.org/)

---

## Requirements

- Python 3.9+
- Any modern browser (Chrome, Firefox, Edge)
