import numpy as np

N_SENSORY = 72   # 6 bands x 12 columns of "road fraction" seen by the camera
N_MOTOR = 64     # neurons whose firing rate drives the car
N_ACTIONS = 2    # steer, gas


def extract_features(obs):
    # CarRacing frame (96x96x3) -> 72 numbers in [0, 1]: how much road is in each patch ahead of the car.
    img = obs[:84].astype(np.int16)  # drop the HUD strip
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    road = (np.abs(r - g) < 12) & (np.abs(g - b) < 12) & (r > 70) & (r < 140)
    patches = road[24:84].reshape(6, 10, 12, 8).mean(axis=(1, 3))
    return patches.reshape(-1).astype(np.float32)


class FlyBrain:
    # Leaky integrate-and-fire network wired with a (fly) connectome.
    # The wiring stays fixed; only the readout from motor neurons to the controls (and one global gain) is trained.

    def __init__(self, W, n_motor=N_MOTOR, seed=0, leak=0.9, threshold=1.0,
                 substeps=5, in_gain=0.5, base_gain=2.5):
        rng = np.random.default_rng(seed)
        self.n = W.shape[0]
        self.Wt = W.T.tocsr()  # W[pre, post] -> input to post = Wt @ spikes
        self.leak, self.threshold, self.substeps = leak, threshold, substeps
        self.in_gain, self.base_gain = in_gain, base_gain
        self.n_motor = n_motor
        self.sensory_idx = rng.choice(self.n, N_SENSORY, replace=False)
        self.motor_idx = rng.choice(self.n, n_motor, replace=False)
        self.n_params = N_ACTIONS * n_motor + N_ACTIONS + 1
        self.set_params(np.zeros(self.n_params))
        self.reset()

    def set_params(self, theta):
        k = N_ACTIONS * self.n_motor
        theta = np.asarray(theta, dtype=np.float32)
        self.Wout = theta[:k].reshape(N_ACTIONS, self.n_motor)
        self.bout = theta[k:k + N_ACTIONS]
        self.gain = float(np.clip(self.base_gain * np.exp(theta[-1]), 1.5, 2.9))

    def reset(self):
        self.v = np.zeros(self.n, np.float32)
        self.s = np.zeros(self.n, np.float32)
        self.rate = np.zeros(self.n_motor, np.float32)

    def step(self, features):
        drive = np.zeros(self.n, np.float32)
        drive[self.sensory_idx] = self.in_gain * features
        for _ in range(self.substeps):
            current = self.gain * (self.Wt @ self.s) + drive
            self.v = self.leak * self.v + current
            self.s = (self.v >= self.threshold).astype(np.float32)
            self.v[self.s > 0] = 0.0
            self.rate = 0.8 * self.rate + 0.2 * self.s[self.motor_idx]
        return self.rate

    def act(self, rate):
        out = self.Wout @ (5.0 * rate) + self.bout
        steer = float(np.tanh(out[0]))
        gas = float(0.5 / (1.0 + np.exp(-out[1])))
        return np.array([steer, gas, 0.0], dtype=np.float32)
