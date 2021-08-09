# GES Algorithm (with C++)

This is the C++ implementation of Greedy Equivalence Search algorithm with C++ & Python frontend.

## How to use
Prerequisite:
- Python 3.8
- PyTorch 1.6+
- CMake 3.12+
- Boost Python & Boost Numpy

Command:
```
pip3 install .
```

## Usage
``` python
from gescpp import run_ges
import numpy as np

# Matrix with shape [time x var]
a = np.random.normal(0, 1, [1000, 10])
graph = run_ges(a)
```

## Reference
- [https://github.com/juangamella/ges.git](https://github.com/juangamella/ges.git)
