from PIL import Image
import numpy as np
from numba import jit, jitclass, double, int32
import numba
import time
import numbers
from functools import reduce
import os
import multiprocessing

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--bounces", type=int, default=3)
parser.add_argument("--timeout", type=int, default=30)

@jit(nopython=True)
def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)




vecspec = [
    ('x', double[:]),               # a simple scalar field
    ('y', double[:]),          # an array field
    ("z", double[:])
]

@jitclass(vecspec)
class vec3:
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def __abs__(self):
        return self.dot(self)
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r

svecspec = [
    ('x', double),               # a simple scalar field
    ('y', double),          # an array field
    ("z", double)
]

@jitclass(svecspec)
class svec3:
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def __abs__(self):
        return self.dot(self)
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r

@jit((double, double, double, int32), nopython=True)
def from_scalars(x, y, z, size):
    return vec3(np.repeat(x, size), np.repeat(y, size), np.repeat(z, size))


rgb = vec3
FARAWAY = 1.0e39            # an implausibly huge distance


spherespec = [
    ('c', svec3.class_type.instance_type),
    ('r', double),
    ('diffuse', svec3.class_type.instance_type),
    ('mirror', double)
]


@jitclass(spherespec)
class Sphere:
    def __init__(self, center, r, diffuse, mirror = 0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, L, O, D, d, scene, bounce, max_bounces):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        E = O
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < max_bounces:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(L, nudged, rayD, scene, bounce + 1, max_bounces) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

@jitclass(spherespec)
class CheckeredSphere:
    def __init__(self, center, r, diffuse, mirror = 0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def light(self, L, O, D, d, scene, bounce, max_bounces):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        E = O
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < max_bounces:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(L, nudged, rayD, scene, bounce + 1, max_bounces) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker

sphere_array = numba.types.Array(Sphere.class_type.instance_type, 1, "A")


@jit((svec3.class_type.instance_type, svec3.class_type.instance_type, vec3.class_type.instance_type, sphere_array, int32, int32) ,nopython=True)
def raytrace(L, O, D, scene, bounce = 0, max_bounces=2):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = np.minimum.reduce(distances, np.array([]))
    color = from_scalars(0, 0, 0, O.shape)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(L, Oc, Dc, dc, scene, bounce, max_bounces)
            color += cc.place(hit)
    return color



@jit((svec3.class_type.instance_type, svec3.class_type.instance_type, vec3.class_type.instance_type, sphere_array, int32), nopython=True)
def do_raytrace(L, E, Q, scene, max_bounces):
    return raytrace(L, E, (Q - E).norm(), scene, max_bounces=max_bounces)

def main(args):
    (w, h) = (args.width, args.height)         # Screen size

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1., 1. / r + .25, 1., -1. / r + .25)
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)


    L = from_scalars(5., 5., -10., x.shape[0])        # Point light position
    E = from_scalars(0., 0.35, -1., x.shape[0])     # Eye position

    print("bhalhlahlahlhlhlhlhlh")
    scene = []
    
    scene.append(Sphere(svec3(.75, .1, 1.), .6, svec3(0., 0., 1.)))
    print("1")
    scene.append(Sphere(svec3(-.75, .1, 2.25), .6, svec3(.5, .223, .5)))
    print("2")
    scene.append(Sphere(svec3(-2.75, .1, 3.5), .6, svec3(1., .572, .184)))
    print("3")
    scene.append(CheckeredSphere(svec3(0.,-99999.5, 0.), 99999, svec3(.75, .75, .75), 0.25))
    print("4")


    t0 = time.time()
    Q = vec3(x, y, np.repeat(0., x.shape[0]))
    color = do_raytrace(L, E, Q, scene, args.bounces)
    print("Took", time.time() - t0)

    new_rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", new_rgb).save("fig.png")

if __name__ == '__main__':
    main(parser.parse_args())