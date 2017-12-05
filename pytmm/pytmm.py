#!/usr/bin/env python
"""
2017-11-30 14:33:01
@author: Paul Reiter
"""
import numpy as np


class LayeredSystem:

    def __init__(self, rho, material):
        self.rho = rho
        self.material = material
        self.layers = []

    def add_layer_in_back(self, layer):
        self.layers.append(layer)

    def add_layer_in_front(self, layer):
        self.layers.insert(0, layer)

    def get_all_kx(self, omega, kx, ky):
        z = self.material.get_z(omega, self.rho)
        kx_list = []
        for layer in self.layers:
            next_z = layer.material.get_z(omega, self.rho)
            kx_list.append(np.sqrt((kx**2 + ky**2)*next_z/z - ky**2))
            kx = kx_list[-1]
            z = next_z
        return kx_list

    def get_amplitudes(self, pressure, velocity, omega, kx):
        a = (velocity * self.material.get_z(omega, self.rho) *
             omega * self.rho / (self.material.porosity * kx) + pressure) / 2
        return a, pressure - a

    def get_rigid_backed_ri(self, omega, kx, ky):
        kx_list = self.get_all_kx(omega, kx, ky)
        pressure, velocity = 1, 0
        for layer, layer_kx in zip(reversed(self.layers), reversed(kx_list)):
            pressure, velocity = layer.transfer(pressure, velocity,
                                                omega, self.rho, layer_kx)
        a, b = self.get_amplitudes(pressure, velocity, omega, kx)
        return np.abs(b)**2 / np.abs(a)**2

    @staticmethod
    def get_kx_ky(angle, k):
        return np.cos(angle) * k, np.sin(angle) * k
