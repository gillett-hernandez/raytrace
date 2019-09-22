#!/usr/bin/env python3
from PIL import Image
import numpy as np
import time
import numbers
from functools import reduce
import os
import multiprocessing
import copy
from core_raytrace import (
    vec3,
    rgb,
    Sphere,
    CheckeredSphere,
    next_highest_divisor,
    raytrace,
    load_and_parse_scene_from_file
)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--bounces", type=int, default=3)
parser.add_argument("--timeout", type=int, default=30)
parser.add_argument("--scenefile", type=str, default="scene.json")


def do_raytrace(L, E, Q, scene, max_bounces):
    print(os.getpid())
    return raytrace(L, E, (Q - E).norm(), scene, max_bounces=max_bounces)
    print(os.getpid(), "done!")


def main(args):
    (w, h) = (args.width, args.height)  # Screen size

    L = vec3(5, 5.0, -10)  # Point light position
    E = vec3(0.0, 0.35, -1.0)  # Eye position

    scene = load_and_parse_scene_from_file(args.scenefile)

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    # fmt:off
    S = (
        -1.0,
        1.0 / r + 0.25,
        1.0,
        -1.0 / r + 0.25
    )
    # fmt:on
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)

    t0 = time.time()
    # N = 4
    N = os.cpu_count()
    # Qs = [vec3(_x, _y, 0) for _x, _y in zip(np.split(x, N), np.split(y, N))]
    points = np.linspace(-0.25, 0.75, N + 1)[::-1]
    Qs = [
        vec3(_x, _y, 0)
        for _x, _y in zip(
            [np.tile(np.linspace(S[0], S[2], w), h // N) for _ in range(N)],
            [
                np.repeat(np.arange(p1, p0, 1 / (h - 1))[::-1], w)
                for p0, p1 in zip(points, points[1:])
            ],
        )
    ]
    # breakpoint()
    with multiprocessing.Pool(processes=N) as pool:
        print(f"starting pool execution on {N} processes")

        print(x.shape, y.shape)
        print("sending starmap order")
        colors = pool.starmap(
            do_raytrace, [copy.deepcopy(tuple([L, E, sub, scene, args.bounces])) for sub in Qs]
        )
        # colors = [pool.apply_async(do_raytrace, copy.deepcopy(tuple([L, E, sub, scene, args.bounces]))) for sub in Qs]
        print("getting results")
        # colors = [res.get(timeout=args.timeout) for res in colors]
        # colors = colors.get(timeout=args.timeout)
        print(type(colors), dir(colors))
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
    # Q = vec3(x, y, 0)
    # color = do_raytrace(L, E, Q, scene, args.bounces)
    print("Took", time.time() - t0)

    new_rgb = [
        Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L")
        for c in color.components()
    ]
    Image.merge("RGB", new_rgb).save("fig.png")


if __name__ == "__main__":
    main(parser.parse_args())
