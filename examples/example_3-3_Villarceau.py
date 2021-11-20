from pysteiner.steiner import Steiner
import pyvista as pv

pltr = pv.Plotter()
Steiner(
  pltr, [3,3], 0.35, 0.2, villarceau=True, rv=0.0025, 
  smooth_shading=True, color="purple", specular=10
)
pltr.show()
