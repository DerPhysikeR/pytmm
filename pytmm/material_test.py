#!/usr/bin/env python
"""
2017-12-01 11:38:21
@author: Paul Reiter
"""
import pytest
from pytmm import Material


def test_material_creation():
    material = Material(1, 2, 3)
    assert material.porosity == 1
    assert material.flow_resistivity == 2
    assert material.tortuosity == 3


@pytest.mark.parametrize('init, para, result', [
    ((1, 0, 1), (1, 1), 1),
    ((1, 0, 3), (1, 1), 3),
    ((2, 2, 0), (1, 1), -4j),
    ((1, 1, 1), (1, 1), 1-1j),
    ((1, 1, 1), (2, 1), 1-.5j),
])
def test_get_z(init, para, result):
    material = Material(*init)
    assert material.get_z(*para) == result
