# Dynamics-preserving graph sparsification

Code for the paper *Dynamics-preserving graph sparsification*. Requires Python ≥ 3.9, NumPy and SciPy.

- `sparsify.py`: DSE and USE sparsification, and effective resistance and random sampling.
- `networks.py`: the twelve networks used in the paper.
- `data/`: the four empirical networks.

```python
import numpy as np
from networks import build
from sparsify import dse_sparsify, use_sparsify, er_sparsify

A = build('human_connectome')                         # weighted adjacency matrix
H = dse_sparsify(A, 0.15)                             # keep 15% of the edges
H_er = er_sparsify(A, 0.15, np.random.default_rng(0))
```
