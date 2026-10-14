# fly-brain-racing

A spiking "fly brain" that learns to drive in Gymnasium's `CarRacing-v3`, trains 24/7, and pushes its progress to GitHub through Composio.

## How it works
- **Brain:** leaky integrate-and-fire neurons wired by a connectome (`flybrain/brain.py`). The wiring is fixed, like a real fly brain.
- **Senses:** the camera frame becomes 72 "how much road is here" signals injected into 72 sensory neurons.
- **Motor:** the firing rate of 64 motor neurons is read out into steering and gas.
- **Learning:** evolution strategies train the readout weights and one global gain (`train.py`). No backprop through spikes.
- **Pushing:** every 30 minutes `push.py` commits `models/best.npz`, `metrics.csv` and `progress.png` to the `training` branch through the Composio GitHub tool.

## Connectome
- By default a **synthetic fly-like graph** is used (sparse, hub-heavy, ~20% inhibitory, 2000 neurons) so everything runs out of the box.
- For the real fly, download a FlyWire connections table (codex.flywire.ai) and run `python train.py --connectome connections.csv --neurons 20000`. Columns are matched by name: pre id, post id, synapse count, optional neurotransmitter.
- Sensory and motor neurons are currently picked at random. Next step: pick them by cell class (visual input neurons, descending neurons).

## Run on NixOS
```
nix-shell
python test_brain.py            # dynamics sanity check
python train.py                 # local training, no pushing
```

## 24/7 with GitHub pushes
```
mkdir -p ~/.config
cat > ~/.config/flybrain.env <<EOF
COMPOSIO_API_KEY=your_key
GITHUB_OWNER=your_github_username
GITHUB_REPO=fly-brain-racing
# COMPOSIO_USER_ID=default
# COMPOSIO_CONNECTED_ACCOUNT_ID=
EOF
mkdir -p ~/.config/systemd/user
cp flybrain-trainer.service ~/.config/systemd/user/
systemctl --user enable --now flybrain-trainer
loginctl enable-linger $USER      # keep running when logged out
journalctl --user -u flybrain-trainer -f
```
Training resumes from `models/latest.npz` after a crash or reboot.
