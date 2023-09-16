from pysteiner.steinerStars import SteinerStars, SteinerStarsMovie
import pyvista as pv

pltr = pv.Plotter(window_size=[512,512])
_ = SteinerStars(
  pltr, [3,2], -0.5, 0.8
)
pltr.set_position([5,5,6])
pltr.show()

SteinerStarsMovie(
        [3, 2], 
        phi = -0.5,
        nframes = 30,
        cameraPosition = [5,5,6])
