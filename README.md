# Dynamics-preserving graph sparsification

Code for the paper *Dynamics-preserving graph sparsification*. Requires Python ≥ 3.9, NumPy and SciPy.

- `sparsify.py`: DSE and USE sparsification, and effective resistance and random sampling.
- `networks.py`: the twelve networks used in the paper.
- `data/`: the four empirical networks.

Example usage:

```python
import numpy as np
from networks import build
from sparsify import dse_sparsify, use_sparsify, er_sparsify

A = build('human_connectome')                         # weighted adjacency matrix
H = dse_sparsify(A, 0.15)                             # keep 15% of the edges
H_er = er_sparsify(A, 0.15, np.random.default_rng(0))
```

## Data

`data/` holds the four empirical networks, from these sources:

- Macaque cortex, *C. intestinalis* and *C. elegans* connectomes: the [Netzschleuder](https://networks.skewed.de)
  entries `macaque_neural`, `cintestinalis` and `celegansneural` (Peixoto 2020), from Honey et al. 2007;
  Ryan, Lu & Meinertzhagen 2016; and Watts & Strogatz 1998 (data of White et al. 1986).
- Human connectome: `weights.txt` of `connectivity_66.zip` in The Virtual Brain data package
  (Sanz Leon et al. 2013), the regions of Hagmann et al. 2008.

The MIT license covers the code only. The four networks in `data/` keep their original terms; please cite
their sources when using them.

## License

The code is released under the MIT License (see `LICENSE`).
