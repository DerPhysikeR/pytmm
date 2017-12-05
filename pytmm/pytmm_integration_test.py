#!/usr/bin/env python
"""
2017-12-05 09:39:53
@author: Paul Reiter
"""
import numpy as np
import pytest
from pytmm import LayeredSystem, Layer, Material


def single_layer_ri(d, sigma, k, ka):
    return np.abs((1 - 1j*sigma*np.tan(ka*d)*k/ka) /
                  (1 + 1j*sigma*np.tan(ka*d)*k/ka))**2


def gapped_layer_ri(d, gap, sigma, k, ka):
    fact = sigma*k/ka
    return np.abs((-1j*fact**2*np.sin(ka*d) +
                   fact*np.cos(ka*d)*(1-1j*np.tan(k*gap)) -
                   np.tan(k*gap)*np.sin(ka*d)) /
                  (1j*fact**2*np.sin(ka*d) +
                   fact*np.cos(ka*d)*(1+1j*np.tan(k*gap)) -
                   np.tan(k*gap)*np.sin(ka*d)))**2


@pytest.mark.parametrize('frequency', np.arange(100, 5000, 100))
@pytest.mark.parametrize('thickness, material', [
    (.1, (.99, 10000, 1.4)),
    (.04, (.95, 30000, 1.55)),
])
def test_single_layer_normal_ri(frequency, thickness, material):
    omega = 2*np.pi*frequency
    k = omega/343
    ka = k * np.sqrt(Material(*material).get_z(omega, 1.205))
    reference = single_layer_ri(thickness, material[0], k, ka)
    ls = LayeredSystem(1.205, Material(1, 0, 1))
    ls.add_layer_in_back(Layer(thickness, Material(*material)))
    to_test = ls.get_rigid_backed_ri(omega, *LayeredSystem.get_kx_ky(0, k))
    assert np.abs(reference - to_test) < 1e-10


@pytest.mark.parametrize('frequency', np.arange(100, 5000, 100))
@pytest.mark.parametrize('thickness, gap, material', [
    (.1, 0, (.99, 10000, 1.4)),
    (.04, .059, (.96, 30000, 1.55)),
])
def test_airgaped_layer_normal_ri(frequency, thickness, gap, material):
    omega = 2*np.pi*frequency
    k = omega/343
    ka = k * np.sqrt(Material(*material).get_z(omega, 1.205))
    reference = gapped_layer_ri(thickness, gap, material[0], k, ka)
    ls = LayeredSystem(1.205, Material(1, 0, 1))
    ls.add_layer_in_back(Layer(thickness, Material(*material)))
    ls.add_layer_in_back(Layer(gap, Material(1, 0, 1)))
    to_test = ls.get_rigid_backed_ri(omega, *LayeredSystem.get_kx_ky(0, k))
    assert np.abs(reference - to_test) < 1e-10


@pytest.mark.parametrize('frequency', np.arange(100, 5000, 500))
@pytest.mark.parametrize('angle', np.linspace(0, np.pi/2, 10)[:-1])
@pytest.mark.parametrize('kappa', [.5, 1, 2])
def test_single_layer_fully_reflective_oblique(frequency, angle, kappa):
    thickness = 1
    omega = 2*np.pi*frequency
    k = omega/343
    ls = LayeredSystem(1.205, Material(1, 0, 1))
    ls.add_layer_in_back(Layer(thickness, Material(1, 0, kappa)))
    to_test = ls.get_rigid_backed_ri(omega, *LayeredSystem.get_kx_ky(angle, k))
    assert np.abs(1 - to_test) < 1e-10


@pytest.mark.parametrize('frequency', np.arange(100, 5000, 500))
@pytest.mark.parametrize('angle', np.linspace(0, np.pi/2, 10)[:-1])
@pytest.mark.parametrize('kappa1', [.5, 1, 2])
@pytest.mark.parametrize('kappa2', [.5, 1, 2])
def test_multi_layer_fully_reflective_oblique(frequency, angle, kappa1,
                                              kappa2):
    thickness = 1
    omega = 2*np.pi*frequency
    k = omega/343
    ls = LayeredSystem(1.205, Material(1, 0, 1))
    ls.add_layer_in_back(Layer(thickness, Material(1, 0, kappa1)))
    ls.add_layer_in_back(Layer(thickness, Material(1, 0, kappa2)))
    to_test = ls.get_rigid_backed_ri(omega, *LayeredSystem.get_kx_ky(angle, k))
    assert np.abs(1 - to_test) < 1e-10
