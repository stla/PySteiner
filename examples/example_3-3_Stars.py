from pysteiner.steinerStars import SteinerStars, SteinerStarsMovie
import pyvista as pv

pltr = pv.Plotter(window_size=[512,512])
_ = SteinerStars(
  pltr, [3,3], 0.15, 0.3
)
pltr.set_position([5,0,6])
pltr.show()

SteinerStarsMovie(
        [3, 3], 
        phi = 0.15,
        nframes = 30,
        cameraPosition = [5,0,6],
        bgcolor = [0.21,0.22,0.25])
