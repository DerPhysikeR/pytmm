# pytmm

A python implementation of the transfer matrix method for acoustics.
It allows the calculation of the reflection index for arbitrary systems of plane absorbing layers and for arbitrary angles of incidence.

## Installation

Clone this repository, `cd` into it and execute:
  
```python
python setup.py develop
```

## Usage

The following code snippet calculates the reflection index of a typical absorber with a thickness of 10cm for a normally impinging sound wave.

```python
import numpy
from pytmm import LayeredSystem, Material, Layer

ls = LayeredSystem(1.205, Material(1, 0, 1))
ls.add_layer_in_back(Layer(.1, Material(.99, 10000, 1.4)))

omega = 2*np.pi*frequency
k = omega/343

ls.get_rigid_backed_ri(omega, *LayeredSystem.get_kx_ky(0, k))
```
