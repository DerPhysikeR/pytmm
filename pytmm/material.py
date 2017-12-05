#!/usr/bin/env python
"""
2017-11-30 16:01:32
@author: Paul Reiter
"""


class Material:

    def __init__(self, porosity, flow_resistivity, tortuosity):
        self.porosity = porosity
        self.flow_resistivity = flow_resistivity
        self.tortuosity = tortuosity

    def get_z(self, omega, rho):
        return (self.tortuosity -
                1j*self.flow_resistivity*self.porosity/(omega*rho))
