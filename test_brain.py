import numpy as np

from flybrain.brain import FlyBrain, N_SENSORY
from flybrain.connectome import synthetic_fly

W = synthetic_fly(2000)
brain = FlyBrain(W)
rng = np.random.default_rng(1)
rates = []
for t in range(200):
    feats = (rng.random(N_SENSORY) > 0.5).astype(np.float32)
    rates.append(brain.step(feats).mean())
rates = np.array(rates)
print("neurons:", brain.n, "synapses:", W.nnz)
print("mean motor rate:", float(rates[50:].mean()), "max:", float(rates.max()))
print("action:", brain.act(brain.rate))
assert np.isfinite(rates).all()
assert 0.0 < rates[50:].mean() < 0.9, "network is silent or saturated - tune base_gain / in_gain"
print("OK")
