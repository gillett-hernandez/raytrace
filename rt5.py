#!/usr/bin/env python3
from PIL import Image
import numpy as np
import time
import numbers
import math
from functools import reduce
import os
import multiprocessing
import copy

import argparse

from core_raytrace import *

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--bounces", type=int, default=3)
parser.add_argument("--timeout", type=int, default=30)
parser.add_argument("--processes", type=int, default=None)



def main(args):
    t_1 = time.time()
    (w, h) = (args.width, args.height)  # Screen size

    L = vec3(5, 5.0, -5.0)  # Point light position
    E = vec3(0.0, 0.35, -1.0)  # Eye position
    S = vec3(0.0, 0.15, 0.0)  # Viewport center position
    S_SIZE = (2, 2 * h / w)  # Viewport size in world coordinates

    # with eye at vec3(0.0, 0.35, -1.0),
    # x is left right,
    # y is up down
    # z is close far.
    scene = [
        Sphere(vec3(0.625, 0.1, 0.5), 0.6, rgb(0.0, 0.0, 0.0), 1.0),
        Sphere(vec3(-0.625, 0.1, 0.5), 0.6, rgb(0.0, 0.0, 0.0), 1.0),
        CheckeredSphere(vec3(0, -99999.5, 0), 99999, rgb(0.75, 0.75, 0.75), 0.25),
    ]

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    # fmt:off
    S = (
        S.x - S_SIZE[0] / 2,
        S.y + S_SIZE[1] / 2,
        S.x + S_SIZE[0] / 2,
        S.y - S_SIZE[1] / 2,
        S.z
    )
    # fmt:on
    # x = np.tile(np.linspace(S[0], S[2], w), h)
    # y = np.repeat(np.linspace(S[1], S[3], h), w)
    # print(x.shape, y.shape)

    t0 = time.time()
    # N = 4
    oldN = os.cpu_count() if args.processes is None else args.processes
    N = next_highest_divisor(h, oldN)
    if N != oldN:
        print(f"rounding up number of processes to be a divisor of {h}. {h} % {N} == {h%N}")
    # Qs = [vec3(_x, _y, 0) for _x, _y in zip(np.split(x, N), np.split(y, N))]

    # points = np.linspace(S[1], S[3], processes + 1)[::-1]
    # Qs = [
    #     vec3(_x, _y, 0)
    #     for _x, _y in zip(
    #         [np.tile(np.linspace(S[0], S[2], w), h // N) for _ in range(N)],
    #         [
    #             np.repeat(np.arange(p1, p0, 1 / (h - 1))[::-1], w)
    #             for p0, p1 in zip(points, points[1:])
    #         ],
    #     )
    # ]
    # breakpoint()
    with multiprocessing.Pool(processes=min(N, os.cpu_count() - 1)) as pool:
        print(f"starting pool execution on {N} processes")

        print("sending starmap order")
        colors = pool.starmap_async(
            do_raytrace,
            [copy.deepcopy(tuple([L, E, S, w, h, scene, args.bounces, i, N])) for i in range(N)],
        )
        # colors = [pool.apply_async(do_raytrace, copy.deepcopy(tuple([L, E, S, w, h, scene, args.bounces, i, N]))) for i in range(N)]
        print("getting results")
        # colors = [res.get(timeout=args.timeout) for res in colors]
        colors = colors.get(timeout=args.timeout)
        t1 = time.time()
        print(f"Took {t1-t0} seconds to compute raytrace and retrieve results from processes")
        # import pdb
        # pdb.set_trace()
        print("merging results")
        common_shape = next(
            c.shape for v in colors for c in v.components() if not isinstance(c, int)
        )
        color = rgb(
            *[
                np.concatenate([c if type(c) != int else np.zeros(common_shape) for c in comp])
                for comp in zip(*[v.components() for v in colors])
            ]
        )
    t2 = time.time()
    print(f"Took {t2-t1} seconds to merge results")
    # Q = vec3(x, y, 0)
    # color = do_raytrace(L, E, Q, scene, args.bounces)

    new_rgb = [
        Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L")
        for c in color.components()
    ]
    Image.merge("RGB", new_rgb).save("fig.png")

    t3 = time.time()
    print(f"Took {t3-t2} seconds to save results")
    print(f"Took {t3-t_1} seconds total to do everything")


if __name__ == "__main__":
    main(parser.parse_args())
