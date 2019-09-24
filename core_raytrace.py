import time
import os
import numbers
import math
from math import cos, sin
from functools import reduce
import numpy as np
import json


def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


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
        return vec3(extract(cond, self.x), extract(cond, self.y), extract(cond, self.z))

    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z


rgb = vec3
FARAWAY = 1.0e39  # an implausibly huge distance


def raytrace(L, O, D, scene, bounce=0, max_bounces=2):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(L, Oc, Dc, dc, scene, bounce, max_bounces)
            color += cc.place(hit)
    return color


class Sphere:
    def __init__(self, center, r, diffuse, mirror=0.5):
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
        M = O + D * d  # intersection point
        N = (M - self.c) * (1.0 / self.r)  # normal
        E = O
        toL = (L - M).norm()  # direction to light
        toO = (E - M).norm()  # direction to ray origin
        nudged = M + N * 0.0001  # M nudged to avoid itself

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


class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker


def next_highest_divisor(N, k):
    t = k
    while N % t != 0:
        t += 1
    return t


def do_raytrace(L, E, S, w, h, scene, max_bounces, task_id, processes):
    print(os.getpid())
    t0 = time.time()
    assert task_id < processes
    points = np.linspace(S[1], S[3], processes + 1)[::-1]
    p0 = points[-task_id - 1]
    p1 = points[-task_id - 2]
    _x = np.tile(np.linspace(S[0], S[2], w), h // processes)
    _y = np.repeat(np.linspace(p0, p1, h // processes), w)
    Q = vec3(_x, _y, S[4])
    rt_result = raytrace(L, E, (Q - E).norm(), scene, max_bounces=max_bounces)
    t1 = time.time()
    print(task_id, os.getpid(), f"done in {t1-t0} seconds!")
    return rt_result


def make_rotation_matrix(theta, axis=0):
    # template = np.array([
    #     [sin(theta), cos(theta)],
    #     [cos(theta), -sin(theta)]
    # ])
    # fmt: off
    if axis == 0:
        return np.array([
            [cos(theta), 0,  -sin(theta)],
            [0,          1,          0 ],
            [sin(theta), 0, cos(theta)]
        ])
    elif axis == 1:
        return np.array([
            [1,         0,           0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)]
        ])
    elif axis == 2:
        return np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0,                    0, 1]
        ])
    # fmt: on


def do_raytrace_v2(L, E, C, S, w, h, scene, max_bounces, task_id, processes):
    print(os.getpid())
    t0 = time.time()
    assert task_id < processes
    points = np.linspace(S[1], S[3], processes + 1)[::-1]
    p0 = points[-task_id - 1]
    p1 = points[-task_id - 2]
    _x = np.tile(np.linspace(S[0], S[2], w), h // processes)
    _y = np.repeat(np.linspace(p0, p1, h // processes), w)
    LR = make_rotation_matrix(S[-2], axis=1)
    UD = make_rotation_matrix(S[-1], axis=0)
    Q = np.stack([_x, _y, np.repeat(C.z, _x.shape)])
    # Q -= E
    Q = np.dot(LR, Q)
    Q = np.dot(UD, Q)
    Q = vec3(*Q)
    Q += C
    E -= C
    E = np.stack(E.components())
    E = np.dot(LR, E)
    E = np.dot(UD, E)
    E = vec3(*E)
    E += C
    # Q += E

    rt_result = raytrace(L, E, (Q - E).norm(), scene, max_bounces=max_bounces)
    t1 = time.time()
    print(task_id, os.getpid(), f"done in {t1-t0} seconds!")
    return rt_result


def translate(obj):
    return {
        "center": vec3(*obj["center"]),
        "r": obj["radius"],
        "diffuse": vec3(*obj["diffuse"]),
        "mirror": obj["mirror"],
    }


def load_and_parse_scene_from_file(filepath):
    with open(filepath, "r") as fd:
        data = json.load(fd)

    objects = []
    for obj in data["objects"]:
        _type = obj["type"]
        del obj["type"]
        if _type == "Sphere":
            objects.append(Sphere(**translate(obj)))
        elif _type == "CheckeredSphere":
            objects.append(CheckeredSphere(**translate(obj)))
    return objects
