from math import sqrt, sin, cos
import random
import numpy as np
import pyvista as pv
# pv.global_theme.return_cpos = True


def Sphere(center, radius):
    return pv.Sphere(radius, np.append(center, 0))
  
def Circle(center, radius, normal, r):
    polygon = pv.Polygon(center, radius, normal, n_sides=360)
    pts0 = polygon.points
    pts = np.vstack((pts0, pts0[0,:]))
    spline = pv.Spline(pts, 1000)
    return spline.tube(radius=r)


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


# plane passing by points p1, p2, p3 #
def plane3pts(p1, p2, p3):
    xcoef = (p1[1] - p2[1]) * (p2[2] - p3[2]) - (p1[2] - p2[2]) * (p2[1] - p3[1])
    ycoef = (p1[2] - p2[2]) * (p2[0] - p3[0]) - (p1[0] - p2[0]) * (p2[2] - p3[2])
    zcoef = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p1[1] - p2[1]) * (p2[0] - p3[0])
    offset = p1[0] * xcoef + p1[1] * ycoef + p1[2] * zcoef
    return np.array([xcoef, ycoef, zcoef, offset])


# center, radius and normal of the circle passing by three points #
def circleCenterAndRadius(p1, p2, p3):
    p12 = (p1 + p2) / 2
    p23 = (p2 + p3) / 2
    v12 = p2 - p1
    v23 = p3 - p2
    plane = plane3pts(p1, p2, p3)
    A = np.column_stack((plane[0:3], v12, v23))
    b = np.array([plane[3], np.vdot(p12, v12), np.vdot(p23, v23)])
    center = np.matmul(np.linalg.inv(np.transpose(A)), b)
    r = np.linalg.norm(p1 - center)
    normal = plane[0:3] / np.linalg.norm(plane[0:3])
    return dict(center=center, radius=r, normal=normal)


# parametrization of Villarceau circles ####
def villarceau_point(mu, a, c, theta, psi, epsilon):
    b = sqrt(a*a-c*c)
    bb = b*sqrt(mu*mu-c*c)
    bb2 = b*b*(mu*mu-c*c)
    denb1 = c*(a*c-mu*c+c*c-a*mu-bb)
    b1 = (a*mu*(c-mu)*(a+c)-bb2+c*c+bb*(c*(a-mu+c)-2*a*mu))/denb1
    denb2 = c*(a*c-mu*c-c*c+a*mu+bb)
    b2 = (a*mu*(c+mu)*(a-c)+bb2-c*c+bb*(c*(a-mu-c)+2*a*mu))/denb2
    omegaT = (b1+b2)/2
    d = (a-c)*(mu-c)+bb
    r = c*c*(mu-c)/((a+c)*(mu-c)+bb)/d
    R = c*c*(a-c)/((a-c)*(mu+c)+bb)/d
    omega2 = (a*mu + bb)/c
    sign = 1 if epsilon > 0 else -1
    f1 = -sqrt(R*R-r*r)*sin(theta)
    f2 = sign*(r+R*cos(theta))
    x1 = f1*cos(psi) + f2*sin(psi) + omegaT
    y1 = f1*sin(psi) - f2*cos(psi)
    z1 = r*sin(theta)
    den = (x1-omega2)**2+y1*y1+z1*z1
    return np.array([omega2 + (x1-omega2)/den, y1/den, z1/den])

# Villarceau circle as mesh (tube) ####
def vcircle(mu, a, c, r, psi, sign, shift):
    p1 = villarceau_point(mu, a, c, 0, psi, sign) + shift
    p2 = villarceau_point(mu, a, c, 2, psi, sign) + shift
    p3 = villarceau_point(mu, a, c, 4, psi, sign) + shift
    circ = circleCenterAndRadius(p1, p2, p3)
    return Circle(circ["center"], circ["radius"], circ["normal"], r)


# n: list of integers, the numbers of spheres at each step
# -1 < phi < 1, phi != 0
def Steiner(
    plotter, n, phi, shift=0, Center=np.zeros((2)), radius=2, 
    epsilon=0.004, cyclide=True, villarceau=False, rv=0.02, **kwargs
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
        
        epsilon -- small positive float to reduce the radii of the balls
        
        cyclide -- whether to plot the enveloping cyclides
        
        villarceau -- whether to plot the enveloping Villarceau circles
        
        rv -- Villarceau circles actually are thin tori; this parameter is the minor radius

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
    if (cyclide or villarceau) and depth == 1:
        circle = iotaCircle(I-Center, k, np.array([0,0]), CRadius - CSide)
        mu = (radius - circle["radius"]) / 2
        a = (radius + circle["radius"]) / 2
        c = (circle["center"][0] - O1x) / 2
        pt = Center + circle["center"] / 2
        translation = (pt[0]-O1x/2, pt[1], 0)
        if cyclide:
            mesh = cyclideMesh(mu, a, c)
            mesh.translate(translation)
            actr = random.randint(0, 1e16)
            actors.append(actr)
            _ = plotter.add_mesh(
                mesh, color="#FFFF00", opacity=0.2, specular=4, name=f"actor_{actr}"
            )
        if villarceau:
            translation = np.asarray(translation)
            psi_ = np.linspace(0, 2*np.pi, 13)[:12]
            for psi in psi_:
                vill1 = vcircle(mu, a, c, rv, psi, -1, translation)
                _ = plotter.add_mesh(vill1, color="red")
                vill2 = vcircle(mu, a, c, rv, psi, 1, translation)
                _ = plotter.add_mesh(vill2, color="red")
    for i in range(int(m)):
        beta = (i + 1 + shift) * 2 * np.pi / m
        pti = vec2(CRadius * np.cos(beta), CRadius * np.sin(beta)) + Center
        cc = iotaCircle(I, k, pti, CSide)
        center = cc["center"] - O1
        r = cc["radius"]
        if depth == 1:
            sph = Sphere(center, r - epsilon)
            actr = random.randint(0, 1e16)
            actors.append(actr)
            _ = plotter.add_mesh(sph, name=f"actor_{actr}", **kwargs)
        elif depth > 1:
            actrs = Steiner(
              plotter, tail(n), phi, -shift, center, r, epsilon, 
              cyclide, villarceau, rv, **kwargs
            )
            actors = actors + actrs
    return actors

# plotter = pv.Plotter()
# _ = Steiner(
#     plotter,
#     [3, 3, 5],
#     phi=0.25,
#     shift=0.4,  # Center=vec2(5,6), radius=5,
#     smooth_shading=True,
#     specular=3,
#     color="purple"
#     )
# #print(plotter.camera_position)
# #plotter.set_position([0, 0, 13])
# plotter.show()

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
