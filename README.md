# Neural Network for Swept Volume Approximation
This is a repository containing the GPU-track code for the first project in UNM's CS591: Deep Reinforcement Learning.

## Setup
Requires Python 3.14. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

Place `swept_volume_data.npz` in the project root.

To verify training works, run:

``python3 neural_network.py --quick-test``

## Python Version Information
Python 3.14.3, PyTorch 2.10.0, Matplotlib 3.10.8

### Credits
Charles Omaoeng - 2026