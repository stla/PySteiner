from pysteiner.steiner import Steiner, SteinerMovie
import pyvista as pv

pltr = pv.Plotter()
Steiner(
  pltr, [3,3], 0.05, 0, cyclide=False, villarceau=True, rv=0.025, 
  smooth_shading=True, color="purple", specular=0.8
)
pltr.show()

SteinerMovie(
        [3, 3], 
        phi = 0.05,
        cyclide = False,
        villarceau = True,
        rv = 0.015, 
        nframes = 30,
        cameraPosition = [6,0,6],
        bgcolor = [0.21,0.22,0.25], 
        smooth_shading = True, specular = 5, color = "#B12A90FF")
