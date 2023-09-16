import functools
import random
import numpy as np
import pyvista as pv


def f_icosahedral_star(x, y, z, a):
    u = x**2 + y**2 + z**2
    v = (
        -z
        * (2 * x + z)
        * (
            x**4
            - x**2 * z**2
            + z**4
            + 2 * (x**3 * z - x * z**3)
            + 5 * (y**4 - y**2 * z**2)
            + 10 * (x * y**2 * z - x**2 * y**2)
        )
    )
    return (1 - u) ** 3 + a * u**3 + a * v


@functools.cache
def Star0():
    # generate data grid for computing the values
    X, Y, Z = np.mgrid[(-0.75):0.75:300j, (-0.75):0.75:300j, (-0.75):0.75:300j]
    # create a structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    # compute and assign the values
    a = -1000
    values = f_icosahedral_star(X, Y, Z, a)
    grid.point_data["values"] = values.ravel(order="F")
    isosurf = grid.contour(isosurfaces=[0])
    mesh = isosurf.extract_geometry().scale(4.0/3.0)
    dists = np.linalg.norm(mesh.points, axis=1)
    mesh['dist'] = (dists - dists.min()) / (dists.max() - dists.min())
    return mesh


# image of circle (center, radius) by the inversion
#   with center c and power k
def iotaCircle(c, k, center, radius):
    r = np.sqrt(np.abs(k))
    z1 = np.sign(k) * (center - c) / r
    D1 = radius * radius / r / r - np.vdot(z1, z1)
    z2 = -z1 / D1
    R2 = np.sqrt(np.vdot(z2, z2) + 1 / D1)
    return dict(center=r * z2 + c, radius=r * R2)


def vec2(x, y):
    return np.array([x, y])


def tail(x):
    y = x.copy()
    del y[0]
    return y


# n: list of integers, the numbers of stars at each step
# -1 < phi < 1, phi != 0
def SteinerStars(
    plotter, n, phi, shift=0, Center=np.zeros((2)), radius=2, **kwargs
):
    """
    Add a nested Steiner chain to a pyvista plotter region.

    Parameters:
        plotter -- the pyvista plotter region

        n -- list of integers, the number of balls at each level

        phi -- controls the shape of the figure

        shift -- kind of rotation angle

        Center -- location parameter of the figure

        radius -- scale parameter of the figure
        
        kwargs -- named arguments passed to `add_mesh`, e.g. color="red"

    """
    if not "actors" in locals():
        actors = []
    depth = len(n)
    invphi = 1 / phi
    roverphi = radius * invphi
    I = vec2(roverphi, 0) + Center
    k = radius * radius * (1 - invphi * invphi)
    m = n[0]
    sine = np.sin(np.pi / m)
    Coef = 1 / (1 + sine)
    O1x = 2 * roverphi
    O1 = vec2(O1x, 0.0)
    CRadius = Coef * radius
    CSide = CRadius * sine
    for i in range(int(m)):
        beta = (i + 1 + shift) * 2 * np.pi / m
        pti = vec2(CRadius * np.cos(beta), CRadius * np.sin(beta)) + Center
        cc = iotaCircle(I, k, pti, CSide)
        center = cc["center"] - O1
        r = cc["radius"]
        if depth == 1:
            star = Star0().scale(r).translate(np.append(center, 0))
            actr = random.randint(0, 1e16)
            actors.append(actr)
            _ = plotter.add_mesh(
                    star, 
                    name=f"actor_{actr}", 
                    scalars = star['dist'],
                    smooth_shading=True,
                    specular=0.2,
                    cmap="turbo",
                    log_scale=False,
                    show_scalar_bar=False,
                    flip_scalars=False,
                    **kwargs
                )
        elif depth > 1:
            actrs = SteinerStars(
              plotter, tail(n), phi, -shift, center, r, **kwargs
            )
            actors = actors + actrs
    return actors



def SteinerStarsMovie(
    n,
    phi,
    cameraPosition=[0, 0, 10],
    bgcolor="white",
    nframes=50,
    gifpath=None,
    **kwargs,
):
    """
    Add a nested Steiner chain to a pyvista plotter region.

    Parameters:
        n -- list of integers, the number of balls at each level

        phi -- controls the shape of the figure

        cameraPosition -- list of coordinates of the camera position

        bgcolor -- background color

        nframes -- number of frames of the gif

        gifpath -- path to the gif to be created; if `None` (default), a
        file name is generated in the current folder

        kwargs -- named arguments passed to `add_mesh`, e.g. color="red"

    """
    pv.global_theme.background = bgcolor
    plotter = pv.Plotter(notebook=False, off_screen=True)
    if gifpath is None:
        gifpath = f"SteinerStars_{n[0]}"
        for k in tail(n):
            gifpath = f"{gifpath}-{k}"
        gifpath = f"{gifpath}.gif"
    plotter.open_gif(gifpath)
    for shift in np.linspace(0, 1, nframes + 1)[:nframes]:
        s = SteinerStars(
            plotter, n, phi=phi, shift=shift, **kwargs
        )
        plotter.set_position(cameraPosition)
        plotter.write_frame()
        for a in s:
            plotter.remove_actor(f"actor_{a}")
    # Closes and finalizes movie
    plotter.close()


