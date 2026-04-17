"""
Genetic Algorithm engine for network capture strategy optimisation.

Chromosome layout  (4 genes):
  [0] protocol_id   int   0=TCP  1=UDP  2=ICMP  3=ANY
  [1] port_min      int   0 – 65535
  [2] port_max      int   port_min – 65535
  [3] sampling_rate float 0.05 – 1.0   (step 0.05)
"""

import random
import math
import copy
from typing import List, Tuple

# ── constants ──────────────────────────────────────────────────────────────
PROTOCOLS   = ["TCP", "UDP", "ICMP", "ANY"]
POP_SIZE    = 60
GENERATIONS = 40
ELITE_N     = 4          # top individuals carried forward unchanged
TOURNEY_K   = 5          # tournament size
CX_PROB     = 0.80       # crossover probability
MUT_PROB    = 0.25       # per-individual mutation probability
MUT_GENE    = 0.40       # per-gene mutation probability (given mutation occurs)
SR_STEP     = 0.05       # sampling-rate resolution


# ── gene helpers ────────────────────────────────────────────────────────────

def _rand_individual() -> list:
    proto = random.randint(0, 3)
    p_min = random.randint(0, 65000)
    p_max = random.randint(p_min + 1, min(p_min + random.randint(500, 20000), 65535))
    sr    = round(random.choice([i * SR_STEP for i in range(1, 21)]), 2)
    return [proto, p_min, p_max, sr]


def _decode(ind: list) -> dict:
    return {
        "protocol":     PROTOCOLS[ind[0]],
        "port_min":     ind[1],
        "port_max":     ind[2],
        "sampling_rate": round(ind[3], 2),
    }


# ── fitness ─────────────────────────────────────────────────────────────────

def _fitness(ind: list, traffic_profile: dict) -> float:
    """
    Score how well this capture strategy covers the given traffic profile.

    traffic_profile keys:
        dominant_protocol  str   most-seen protocol in traffic
        hot_ports          list  list of (port, hit_count) sorted desc
        total_flows        int
        avg_packet_size    float
    """
    proto, p_min, p_max, sr = ind
    port_range = p_max - p_min                        # 1 – 65535
    proto_name = PROTOCOLS[proto]

    # ── 1. protocol match ────────────────────────────────────────────
    proto_weights = {"TCP": 1.25, "UDP": 1.10, "ICMP": 0.80, "ANY": 1.0}
    dom = traffic_profile.get("dominant_protocol", "TCP")
    match_bonus = 1.30 if proto_name == dom or proto_name == "ANY" else 0.75
    p_score = proto_weights[proto_name] * match_bonus

    # ── 2. port coverage of hot ports ────────────────────────────────
    hot_ports   = traffic_profile.get("hot_ports", [])
    covered     = sum(1 for (p, _) in hot_ports if p_min <= p <= p_max)
    total_hot   = max(len(hot_ports), 1)
    cov_score   = covered / total_hot

    # ── 3. port-range efficiency (penalise both too narrow + too wide) ─
    ideal_range = 5000
    range_ratio = port_range / 65535
    efficiency  = 1.0 - abs(range_ratio - (ideal_range / 65535))
    efficiency  = max(efficiency, 0.05)

    # ── 4. sampling-rate reward (higher = better coverage, diminishing) ─
    sr_score = math.log1p(sr * 9) / math.log1p(9)   # 0→0, 1→1, concave

    # ── 5. port entropy bonus ─────────────────────────────────────────
    entropy_bonus = math.log2(port_range + 2) / 16.0  # max ≈1 at full range

    # ── 6. flow throughput proxy ──────────────────────────────────────
    total_flows = traffic_profile.get("total_flows", 1000)
    throughput  = min(total_flows * sr / 1000, 1.0)

    # ── weighted sum ──────────────────────────────────────────────────
    score = (
        p_score      * 3.0 +
        cov_score    * 4.0 +
        efficiency   * 2.0 +
        sr_score     * 2.5 +
        entropy_bonus* 1.5 +
        throughput   * 2.0
    )
    return round(score, 6)


# ── selection ────────────────────────────────────────────────────────────────

def _tournament(pop: List[list], scores: List[float]) -> list:
    contestants = random.sample(list(zip(pop, scores)), TOURNEY_K)
    return max(contestants, key=lambda x: x[1])[0]


# ── crossover ────────────────────────────────────────────────────────────────

def _crossover(a: list, b: list) -> Tuple[list, list]:
    if random.random() > CX_PROB:
        return copy.copy(a), copy.copy(b)
    point = random.randint(1, 3)
    c1 = a[:point] + b[point:]
    c2 = b[:point] + a[point:]
    # repair port ordering
    for c in (c1, c2):
        if c[1] >= c[2]:
            c[1], c[2] = sorted(random.sample(range(0, 65535), 2))
    return c1, c2


# ── mutation ─────────────────────────────────────────────────────────────────

def _mutate(ind: list) -> list:
    if random.random() > MUT_PROB:
        return ind
    ind = copy.copy(ind)
    for gene_idx in range(4):
        if random.random() < MUT_GENE:
            if gene_idx == 0:
                ind[0] = random.randint(0, 3)
            elif gene_idx == 1:
                ind[1] = random.randint(0, max(0, ind[2] - 1))
            elif gene_idx == 2:
                ind[2] = random.randint(ind[1] + 1, min(ind[1] + 30000, 65535))
            else:
                options = [round(i * SR_STEP, 2) for i in range(1, 21)]
                ind[3]  = random.choice(options)
    return ind


# ── main GA loop ─────────────────────────────────────────────────────────────

def run_ga(traffic_profile: dict) -> Tuple[dict, List[float]]:
    """
    Run the GA and return (best_strategy_dict, fitness_history).
    fitness_history[i] = best fitness at generation i.
    """
    random.seed()  # fresh seed every run

    population = [_rand_individual() for _ in range(POP_SIZE)]
    fitness_history: List[float] = []

    best_ind    = None
    best_score  = -1.0

    for gen in range(GENERATIONS):
        scores = [_fitness(ind, traffic_profile) for ind in population]

        gen_best_score = max(scores)
        gen_best_ind   = population[scores.index(gen_best_score)]
        fitness_history.append(round(gen_best_score, 4))

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_ind   = copy.copy(gen_best_ind)

        # ── elitism ──────────────────────────────────────────────────
        elite_pairs = sorted(zip(scores, population), reverse=True)
        elites = [copy.copy(ind) for _, ind in elite_pairs[:ELITE_N]]

        # ── build next generation ────────────────────────────────────
        next_pop = elites[:]
        while len(next_pop) < POP_SIZE:
            p1 = _tournament(population, scores)
            p2 = _tournament(population, scores)
            c1, c2 = _crossover(p1, p2)
            next_pop.append(_mutate(c1))
            if len(next_pop) < POP_SIZE:
                next_pop.append(_mutate(c2))

        population = next_pop

    return _decode(best_ind), fitness_history
