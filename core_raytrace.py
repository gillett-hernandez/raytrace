import time
import os
import numbers
import math
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

    # def __rmul__(self, other):
    #     return vec3(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self):
        return self.dot(self)

    def __neg__(self):
        return vec3(-self.x, -self.y, -self.z)

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


def raytrace(L, O, D, scene, bounce=0, max_bounces=2, refract=0, max_refractions=2):
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
            cc = s.light(L, Oc, Dc, dc, scene, bounce, max_bounces, refract, max_refractions)
            color += cc.place(hit)
    return color


class Sphere:
    # index of refraction (ior) of 1.5 is air to glass
    def __init__(self, center, r, diffuse, mirror=0.5, refract=0.0, ior=1.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror
        self.refract = refract
        self.index_of_refraction = ior

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

    def do_bounce(self, L, D, N, nudged, scene, bounce, max_bounces, refract, max_refractions):
        # Reflection
        if bounce < max_bounces:
            rayD = (D - N * 2 * D.dot(N)).norm()
            return raytrace(L, nudged, rayD, scene, bounce + 1, max_bounces, refract, max_refractions) * self.mirror
        return vec3(0, 0, 0)

    def do_refraction(self, L, D, N, nudged, scene, bounce, max_bounces, refract, max_refractions):
        # Refraction
        if refract < max_refractions and self.refract > 0:
            NdI = N.dot(D)
            eta_i, eta_t = 1, self.index_of_refraction
            cos_i = np.clip(NdI, -1, 1)
            N = vec3(*[np.where(cos_i >= 0, -c, c) for c in N.components()])
            eta = eta_i/eta_t
            eta = np.where(cos_i < 0, eta, 1/eta)
            cos_i = np.where(cos_i < 0, cos_i, -cos_i)
            # outside of the surface
            # inside the surface. reverse normal
            # this has an effect on the later bounce check too.
            k = 1 - eta**2 * (1 - cos_i**2)
            # if k < 0:
            # total internal reflection
            # do bounce check.
            # self.do_bounce(bounce, max_bounces, L, D, N, nudged, scene, refract, max_refractions)
            # TIR_origin =
            # total_internal_reflection = np.where(k<0, self.do_bounce(L, D, N, nudged, scene, bounce, max_bounces, refract, max_refractions), vec3(0, 0, 0))
            # else:
            rayD = D*eta + N*(eta * cos_i - k**0.5)
            cos_i_b_0 = np.maximum(0, 1 - cos_i * cos_i)
            sint = eta_i / eta_t * cos_i_b_0**0.5
            # if (sint >= 1) {
            #     kr = 1;
            # }
            # else {
            cos_t = np.maximum(0, 1 - sint * sint)**0.5
            cos_i = abs(cos_i)
            Rs = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t))
            Rp = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t))
            kr = (Rs * Rs + Rp * Rp) / 2
            kr = np.where(sint>=1, 1, kr)
            return (raytrace(L, nudged, rayD, scene, bounce, max_bounces, refract+1, max_refractions) * self.refract, kr)
        return (vec3(0, 0, 0), 1)

    def light(self, L, O, D, d, scene, bounce, max_bounces, refract, max_refractions):
        M = O + D * d  # intersection point
        N = (M - self.c) * (1.0 / self.r)  # normal
        E = O
        toL = (L - M).norm()  # direction to light
        toO = (E - M).norm()  # direction to ray origin
        nudged = M + N * 0.0001  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        # light_distances = [s.intersect(nudged, toL) for s in scene]
        # light_nearest = reduce(np.minimum, light_distances)
        # seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * 1.0
        reflection = self.do_bounce(L, D, N, nudged, scene, bounce, max_bounces, refract, max_refractions)

        refraction, kr = self.do_refraction(L, D, N, nudged, scene, bounce, max_bounces, refract, max_refractions)
        color += reflection*kr + refraction*(1-kr)

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * 1.0
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


def do_raytrace_v2(L, E, S, w, h, scene, task_id, processes, max_bounces, max_refractions):
    print(os.getpid())
    t0 = time.time()
    assert task_id < processes
    points = np.linspace(S[1], S[3], processes + 1)[::-1]
    p0 = points[-task_id - 1]
    p1 = points[-task_id - 2]
    _x = np.tile(np.linspace(S[0], S[2], w), h // processes)
    _y = np.repeat(np.linspace(p0, p1, h // processes), w)
    Q = vec3(_x, _y, S[4])
    rt_result = raytrace(L, E, (Q - E).norm(), scene, max_bounces=max_bounces, max_refractions=max_refractions)
    t1 = time.time()
    print(task_id, os.getpid(), f"done in {t1-t0} seconds!")
    return rt_result


def translate(obj):
    return {
        "center": vec3(*obj["center"]),
        "r": obj["radius"],
        "diffuse": vec3(*obj["diffuse"]),
        "mirror": obj["mirror"],
        "refract": obj["refract"]
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
