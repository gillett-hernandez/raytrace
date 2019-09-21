#!/usr/bin/env python3
from PIL import Image
import numpy as np
import time
import numbers
from functools import reduce
import os

from core_raytrace import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--bounces", type=int, default=3)


def do_raytrace(L, E, Q, scene, max_bounces):
    return raytrace(L, E, (Q - E).norm(), scene, max_bounces=max_bounces)


def main(args):
    (w, h) = (args.width, args.height)  # Screen size

    L = vec3(5, 5.0, -10)  # Point light position
    E = vec3(0.0, 0.35, -1.0)  # Eye position

    scene = [
        Sphere(vec3(0.625, 0.1, 0.5), 0.6, rgb(0.0, 0.0, 0.0), 1.0),
        Sphere(vec3(-0.625, 0.1, 0.5), 0.6, rgb(0.0, 0.0, 0.0), 1.0),
        CheckeredSphere(vec3(0, -99999.5, 0), 99999, rgb(0.75, 0.75, 0.75), 0.25),
    ]

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1.0, 1.0 / r + 0.25, 1.0, -1.0 / r + 0.25)
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)

    t0 = time.time()
    Q = vec3(x, y, 0)
    color = do_raytrace(L, E, Q, scene, args.bounces)
    print("Took", time.time() - t0)

    new_rgb = [
        Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L")
        for c in color.components()
    ]
    Image.merge("RGB", new_rgb).save("fig.png")


if __name__ == "__main__":
    main(parser.parse_args())
