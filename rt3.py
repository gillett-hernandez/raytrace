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
    (w, h) = (args.width, args.height)         # Screen size

    L = vec3(5, 5., -10)        # Point light position
    E = vec3(0., 0.35, -1.)     # Eye position

    scene = [
        Sphere(vec3(.75, .1, 1.), .6, rgb(0, 0, 1)),
        Sphere(vec3(-.75, .1, 2.25), .6, rgb(.5, .223, .5)),
        Sphere(vec3(-2.75, .1, 3.5), .6, rgb(1., .572, .184)),
        CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.75, .75, .75), 0.25),
        ]

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1., 1. / r + .25, 1., -1. / r + .25)
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)

    t0 = time.time()
    # N = os.cpu_count()
    # with multiprocessing.Pool(processes=N) as pool:
    #     Qs = [vec3(_x, _y, 0) for _x, _y in zip(np.split(x, N), np.split(y, N))]
    #     print(x.shape, y.shape)
    #     colors = pool.starmap_async(raytrace, [(L, E, (S - E).norm(), scene) for S in Qs])
    #     # colors = [res.get(timeout=args.timeout) for res in colors]
    #     colors = colors.get(timeout=args.timeout)
    #     print(type(colors), dir(colors))
    #     # import pdb
    #     # pdb.set_trace()
    #     common_shape = next(c.shape for v in colors for c in v.components() if not isinstance(c, int))
    #     color = rgb(*[np.concatenate([c if type(c) != int else np.zeros(common_shape) for c in comp]) for comp in zip(*[v.components() for v in colors])])
    Q = vec3(x, y, 0)
    color = do_raytrace(L, E, Q, scene, args.bounces)
    print("Took", time.time() - t0)

    new_rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", new_rgb).save("fig.png")

if __name__ == '__main__':
    main(parser.parse_args())