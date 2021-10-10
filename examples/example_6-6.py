# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 10:40:52 2021

@author: SDL96354
"""

from pysteiner.steiner import SteinerMovie

SteinerMovie(
        [6,6], 
        phi = 0.25,
        nframes = 20,
        cameraPosition = [6,0,6],
        bgcolor = [0.21,0.22,0.25], 
        smooth_shading = True, specular = 5, color = "#B12A90FF")