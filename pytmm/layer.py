#!/usr/bin/env python
"""
2017-11-30 14:00:00
@author: Paul Reiter
"""
import numpy as np


class Layer:

    def __init__(self, thickness, material):
        self.thickness = thickness
        self.material = material

    def get_transfer_matrix(self, omega, rho, kx):
        factor = (self.material.get_z(omega, rho)*omega*rho /
                  (self.material.porosity*kx))
        arg = kx*self.thickness
        return np.array([[np.cos(arg), 1j*factor*np.sin(arg)],
                         [1j*np.sin(arg)/factor, np.cos(arg)]])

    def transfer(self, pressure, velocity, omega, rho, kx):
        new_pressure, new_velocity = \
            np.dot(self.get_transfer_matrix(omega, rho, kx),
                   np.array([[pressure], [velocity]]))
        return new_pressure[0], new_velocity[0]
