import argparse
import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from flybrain.brain import FlyBrain, extract_features
from flybrain.connectome import load_flywire, synthetic_fly

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_brain = None
_env = None
_max_steps = 500


def make_brain(connectome, neurons, seed):
    if connectome:
        W = load_flywire(connectome, neurons)
    else:
        W = synthetic_fly(neurons, seed=seed)
    return FlyBrain(W, seed=seed)


def _init_worker(connectome, neurons, seed, max_steps):
    global _brain, _env, _max_steps
    import gymnasium as gym

    _brain = make_brain(connectome, neurons, seed)
    _env = gym.make("CarRacing-v3")
    _max_steps = max_steps


def rollout(job):
    theta, ep_seed = job
    _brain.set_params(theta)
    _brain.reset()
    obs, _ = _env.reset(seed=int(ep_seed))
    total = 0.0
    for _ in range(_max_steps):
        action = _brain.act(_brain.step(extract_features(obs)))
        obs, reward, terminated, truncated, _ = _env.step(action)
        total += reward
        if terminated or truncated or total < -25:
            break
    return total


def rank_shape(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def make_plot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(open("metrics.csv")))
        gen = [int(r["gen"]) for r in rows]
        plt.figure(figsize=(8, 4))
        plt.plot(gen, [float(r["pop_mean"]) for r in rows], label="population mean", alpha=0.6)
        plt.plot(gen, [float(r["center"]) for r in rows], label="current brain")
        plt.plot(gen, [float(r["best_center"]) for r in rows], label="best so far")
        plt.xlabel("generation")
        plt.ylabel("CarRacing reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig("progress.png", dpi=110)
        plt.close()
    except Exception as exc:  # plotting must never stop training
        print("plot skipped:", exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", default=None, help="FlyWire connections CSV; omit for a synthetic fly-like graph")
    ap.add_argument("--neurons", type=int, default=2000)
    ap.add_argument("--pairs", type=int, default=8, help="antithetic pairs per generation")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push-every", type=int, default=0, help="seconds between GitHub pushes via Composio (0 = never)")
    ap.add_argument("--generations", type=int, default=0, help="0 = run forever")
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True)
    probe = make_brain(args.connectome, args.neurons, args.seed)
    d = probe.n_params
    n_neurons = probe.n
    rng = np.random.default_rng(args.seed + 1)

    theta, gen, best = np.zeros(d), 0, -1e9
    if Path("models/latest.npz").exists():
        ck = np.load("models/latest.npz")
        if int(ck["n_params"]) == d:
            theta, gen, best = ck["theta"], int(ck["gen"]), float(ck["best"])
            print("resumed at generation", gen)
    if not Path("metrics.csv").exists():
        csv.writer(open("metrics.csv", "w")).writerow(["gen", "time", "pop_mean", "pop_max", "center", "best_center", "neurons"])

    m, v, t = np.zeros(d), np.zeros(d), 0
    last_push = time.time()
    start = time.time()

    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.connectome, args.neurons, args.seed, args.max_steps)) as pool:
        while args.generations == 0 or gen < args.generations:
            gen += 1
            eps = rng.standard_normal((args.pairs, d))
            cands = [theta + args.sigma * e for e in eps] + [theta - args.sigma * e for e in eps]
            jobs = [(c, gen) for c in cands] + [(theta, gen), (theta, gen + 10000)]
            fit = np.array(pool.map(rollout, jobs))
            pop, center = fit[:-2], float(fit[-2:].mean())

            s = rank_shape(pop)
            grad = ((s[:args.pairs] - s[args.pairs:]) @ eps) / (2 * args.pairs * args.sigma)
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            theta = theta + args.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

            if center > best:
                best = center
                np.savez("models/best.npz", theta=theta, n_params=d, gen=gen, best=best, neurons=n_neurons)
            np.savez("models/latest.npz", theta=theta, n_params=d, gen=gen, best=best, neurons=n_neurons)
            csv.writer(open("metrics.csv", "a")).writerow(
                [gen, int(time.time() - start), round(float(pop.mean()), 2), round(float(pop.max()), 2),
                 round(center, 2), round(best, 2), n_neurons])
            print("gen %d  pop %.1f  max %.1f  brain %.1f  best %.1f" % (gen, pop.mean(), pop.max(), center, best), flush=True)

            if args.push_every and time.time() - last_push >= args.push_every:
                last_push = time.time()
                try:
                    from push import push_run

                    make_plot()
                    push_run(["models/best.npz", "metrics.csv", "progress.png"],
                             "gen %d: best reward %.1f (%d neurons)" % (gen, best, n_neurons))
                    print("pushed to GitHub", flush=True)
                except Exception as exc:  # a network hiccup must never stop training
                    print("push failed:", exc, flush=True)


if __name__ == "__main__":
    main()
