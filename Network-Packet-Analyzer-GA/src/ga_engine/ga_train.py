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
