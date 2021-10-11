# -*- coding: utf-8 -*-

from pysteiner.steiner import SteinerMovie

SteinerMovie(
        [3,3,5], 
        phi=0.25, 
        nframes = 30,
        cameraPosition = [5,0,6],
        bgcolor=[0.21,0.22,0.25], 
        smooth_shading=True, specular=5, color="#B12A90FF")
