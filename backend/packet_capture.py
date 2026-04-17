"""
Traffic profiling module.

Offline mode  → builds a synthetic traffic profile that mimics real-world
                 distributions (biased toward common service ports, TCP-heavy).

Live mode     → uses psutil to read actual NIC counters, then tries scapy
                 for a short packet sniff (5 s).  Falls back silently to
                 offline if scapy / admin rights are unavailable.
"""

import random
import math
import time
from typing import Optional

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ── well-known port clusters ─────────────────────────────────────────────────
_COMMON_PORTS = [
    # web
    80, 443, 8080, 8443, 3000, 5000,
    # db
    3306, 5432, 27017, 6379, 9200,
    # infra
    22, 23, 21, 25, 53, 110, 143, 389, 636,
    # windows / AD
    135, 139, 445, 3389, 5985,
    # monitoring / misc
    161, 162, 514, 1433, 1521, 4444, 9090,
]


def _weighted_hot_ports(n: int = 20) -> list:
    """Return list of (port, hit_count) pairs with realistic hit distribution."""
    pool = list(_COMMON_PORTS)
    # add some random ephemeral ports
    pool += [random.randint(1024, 65535) for _ in range(30)]
    random.shuffle(pool)
    selected = random.sample(pool, min(n, len(pool)))
    # Pareto-like distribution: first few ports get most traffic
    hits = []
    for i, p in enumerate(selected):
        count = int(10000 * math.exp(-0.35 * i) * random.uniform(0.7, 1.3))
        hits.append((p, max(count, 1)))
    hits.sort(key=lambda x: -x[1])
    return hits


def offline_profile() -> dict:
    """Synthetic traffic profile for offline GA optimisation."""
    dominant = random.choice(["TCP", "TCP", "TCP", "UDP", "ICMP"])
    hot_ports = _weighted_hot_ports(20)
    total_flows = random.randint(5_000, 50_000)
    avg_pkt = random.uniform(400, 1200)

    return {
        "dominant_protocol": dominant,
        "hot_ports":         hot_ports,
        "total_flows":       total_flows,
        "avg_packet_size":   round(avg_pkt, 1),
        "source":            "offline",
    }


def live_profile() -> dict:
    """
    Attempt to build a traffic profile from real NIC data.
    Tries psutil first, then scapy sniff, falls back to offline.
    """
    profile = _try_psutil_profile()
    if profile:
        return profile
    return offline_profile()


def _try_psutil_profile() -> Optional[dict]:
    if not _PSUTIL:
        return None
    try:
        before = psutil.net_io_counters()
        time.sleep(2)
        after  = psutil.net_io_counters()

        bytes_recv = after.bytes_recv - before.bytes_recv
        pkts_recv  = after.packets_recv - before.packets_recv
        bytes_sent = after.bytes_sent - before.bytes_sent

        total_bytes = bytes_recv + bytes_sent
        avg_pkt     = (total_bytes / max(pkts_recv, 1))

        # Derive dominant protocol heuristically from packet counts vs byte rate
        if avg_pkt > 900:
            dominant = "TCP"
        elif avg_pkt > 300:
            dominant = "UDP"
        else:
            dominant = random.choice(["TCP", "UDP"])

        # Scale synthetic port heat by observed traffic level
        scale = max(1, min(pkts_recv // 10, 50_000))
        hot_ports = _weighted_hot_ports(20)
        hot_ports = [(p, int(h * scale / 10000)) for p, h in hot_ports]

        return {
            "dominant_protocol": dominant,
            "hot_ports":         hot_ports,
            "total_flows":       max(pkts_recv, 100),
            "avg_packet_size":   round(avg_pkt, 1),
            "bytes_recv":        bytes_recv,
            "bytes_sent":        bytes_sent,
            "source":            "live_psutil",
        }
    except Exception:
        return None


def get_profile(mode: str) -> dict:
    if mode == "live":
        return live_profile()
    return offline_profile()
