import random
import numpy as np
import pyvista as pv

# pv.global_theme.return_cpos = True


def Sphere(center, radius):
    return pv.Sphere(radius, np.append(center, 0))


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


def cyclideMesh(mu, a, c):
    angle = np.linspace(0, 2*np.pi, 100) 
    u, v = np.meshgrid(angle, angle)
    b = np.sqrt(a * a - c * c)
    cosu = np.cos(u)
    cosv = np.cos(v)
    h = a - c * cosu * cosv
    x = (mu * (c - a * cosu * cosv) + b * b * cosu) / h
    y = (b * np.sin(u) * (a - mu * cosv)) / h
    z = b * np.sin(v) * (c * cosu - mu) / h
    grid = pv.StructuredGrid(x, y, z)
    return grid.extract_geometry().clean(tolerance=1e-6)


# n: list of integers, the numbers of spheres at each step
# -1 < phi < 1, phi != 0
def Steiner(
    plotter, n, phi, shift=0, Center=np.zeros((2)), radius=2, epsilon=0.005, **kwargs
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
    if depth == 1:
        circle = iotaCircle(I-Center, k, np.array([0,0]), CRadius - CSide)
        mu = (radius - circle["radius"]) / 2
        a = (radius + circle["radius"]) / 2
        c = (circle["center"][0] - O1x) / 2
        pt = Center + circle["center"] / 2
        mesh = cyclideMesh(mu, a, c)
        mesh.translate((pt[0]-O1x/2, pt[1], 0))
        _ = plotter.add_mesh(mesh, color="#FFFF00", opacity=0.2)
    for i in range(int(m)):
        beta = (i + 1 + shift) * 2 * np.pi / m
        pti = vec2(CRadius * np.cos(beta), CRadius * np.sin(beta)) + Center
        cc = iotaCircle(I, k, pti, CSide)
        center = cc["center"] - O1
        r = cc["radius"]
        if depth == 1:
            sph = Sphere(center, r - epsilon)
            a = random.randint(0, 1e15)
            actors.append(a)
            _ = plotter.add_mesh(sph, name=f"actor_{a}", **kwargs)
        elif depth > 1:
            actrs = Steiner(plotter, tail(n), phi, -shift, center, r, epsilon, **kwargs)
            actors = actors + actrs
    return actors


plotter = pv.Plotter()
_ = Steiner(
    plotter,
    [3, 3, 5],
    phi=0.25,
    shift=0.4,  # Center=vec2(5,6), radius=5,
    smooth_shading=True,
    specular=3,
    color="purple"
    )
#print(plotter.camera_position)
plotter.set_position([0, 0, 13])
plotter.show()

# plotter = pv.Plotter()
# def steiner_shift(shift):
#     plotter.clear()
#     Steiner(
#         plotter,
#         [3, 3, 5],
#         phi=0.2,
#         shift=shift,
#         smooth_shading=True,
#         specular=3,
#         color="purple",
#     )
# slider = plotter.add_slider_widget(steiner_shift, [-1,1], event_type="end")
# plotter.show()


def SteinerMovie(
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
        gifpath = f"Steiner_{n[0]}"
        for k in tail(n):
            gifpath = f"{gifpath}-{k}"
        gifpath = f"{gifpath}.gif"
    plotter.open_gif(gifpath)
    for shift in np.linspace(0, 1, nframes + 1)[:nframes]:
        s = Steiner(plotter, n, phi=phi, shift=shift, **kwargs)
        plotter.set_position(cameraPosition)
        plotter.write_frame()
        for a in s:
            plotter.remove_actor(f"actor_{a}")
    # Closes and finalizes movie
    plotter.close()


# Steiner_gif([5,4,3], 0.2, bgcolor="honeydew", smooth_shading=True, specular=3, color="purple")

# def SteinerMovie(
#     n,
#     phi,
#     cameraPosition=[0, 0, 20],
#     bgcolor="white",
#     nframes=50,
#     duration=20,
#     gifpath=None,
#     **kwargs,
# ):
#     pv.global_theme.background = bgcolor
#     if gifpath == None:
#         gifpath = f"Steiner_{n[0]}"
#         for k in tail(n):
#             gifpath = f"{gifpath}-{k}"
#         gifpath = f"{gifpath}.gif"
#     tmpdir = tempfile.TemporaryDirectory()
#     tmpfiles = []
#     for shift in np.linspace(0, 1, nframes + 1)[:nframes]:
#         plotter = pv.Plotter(notebook=False, off_screen=True)
#         tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".png", dir=tmpdir.name, delete=False)
#         tmpfiles.append(tmpfile.name)
#         tmpfile.close()
#         os.unlink(tmpfile.name)
#         _ = Steiner(plotter, n, phi=phi, shift=shift, **kwargs)
#         plotter.set_position(cameraPosition)
#         plotter.show(screenshot=tmpfile.name, return_img=False)
#     #     plotter.close()
#     img, *imgs = [Image.open(f) for f in tmpfiles]
#     img.save(gifpath, format="GIF", append_images=imgs,
#          save_all=True, duration=duration, loop=0)
#
# SteinerMovie([3,3,5], 0.2, nframes=50, duration=20, smooth_shading=True, specular=3, color="lime")
