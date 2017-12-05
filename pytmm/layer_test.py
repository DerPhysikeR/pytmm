#!/usr/bin/env python
"""
2017-12-01 12:11:06
@author: Paul Reiter
"""
import pytest
import numpy as np
from pytmm import Material, Layer


def test_layer_creation():
    layer = Layer(1, 2)
    assert layer.thickness == 1
    assert layer.material == 2


@pytest.mark.parametrize('thickness, material, para, result', [
    (np.pi, (1, 0, 1), (1, 1, 1), np.array([[-1, 0], [0, -1]])),
    (0, (1, 0, 1), (1, 1, 1), np.array([[1, 0], [0, 1]])),
])
def test_get_transfer_matrix(thickness, material, para, result):
    layer = Layer(thickness,  Material(*material))
    np.testing.assert_almost_equal(layer.get_transfer_matrix(*para), result)


def test_transfer():
    layer = Layer(1, 1)
    layer.get_transfer_matrix = lambda x, y, z: np.array([[1, 1], [0, 1]])
    assert all([a == b for a, b in zip(layer.transfer(1, 2, 1, 1, 1), (3, 2))])
