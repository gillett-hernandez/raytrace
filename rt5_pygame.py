#!/usr/bin/env python3
import time
import os
import multiprocessing
import copy
import argparse

import numpy as np
import pygame
from pygame.locals import *


from core_raytrace import *


parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=800)
parser.add_argument("--height", type=int, default=640)
parser.add_argument("--bounces", type=int, default=3)
parser.add_argument("--timeout", type=int, default=30)
parser.add_argument("--processes", type=int, default=None)


def main(args):
    pygame.init()
    SIZE = (w, h) = (args.width, args.height)  # Screen size
    display = pygame.display.set_mode(SIZE)

    screen = pygame.Surface(SIZE)

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
    compute_viewport = lambda S: (
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
        print(
            f"rounding up number of processes to be a divisor of {h}. {h} % {N} == {h%N}"
        )

    first_execution = True
    with multiprocessing.Pool(processes=N) as pool:
        while True:
            invalidated = False
            for e in pygame.event.get():
                if (e.type == KEYDOWN and e.key == K_ESCAPE) or e.type == QUIT:
                    return
                if e.type == KEYDOWN:
                    delta = 0.1
                    y = 0
                    x = 0
                    z = 0
                    r = 0
                    bounce_delta = 0
                    if e.key == K_w:
                        y += 1
                    elif e.key == K_s:
                        y -= 1
                    elif e.key == K_a:
                        x -= 1
                    elif e.key == K_d:
                        x += 1
                    elif e.key == K_UP:
                        z += 1
                    elif e.key == K_DOWN:
                        z -= 1
                    elif e.key == K_b:
                        bounce_delta += 1
                    elif e.key == K_n:
                        bounce_delta -= 1

                    if y != 0 or z != 0 or x != 0 or bounce_delta:
                        invalidated = True

                    S.x += x * delta
                    S.y += z * delta
                    S.z += y * delta
                    E.x += x * delta
                    E.y += z * delta
                    E.z += y * delta
                    args.bounces += bounce_delta
                    # TODO: figure out how to rotate camera + eye positions.
            if invalidated or first_execution:
                first_execution = False
                print(f"starting pool execution on {N} processes")

                print("sending starmap order")
                colors = pool.starmap(
                    do_raytrace,
                    [
                        copy.deepcopy(
                            tuple(
                                [
                                    L,
                                    E,
                                    compute_viewport(S),
                                    w,
                                    h,
                                    scene,
                                    args.bounces,
                                    i,
                                    N,
                                ]
                            )
                        )
                        for i in range(N)
                    ],
                )
                # colors = [pool.apply_async(do_raytrace, copy.deepcopy(tuple([L, E, sub, scene, args.bounces]))) for sub in Qs]
                print("getting results")
                t1 = time.time()
                print(
                    f"Took {t1-t0} seconds to compute raytrace and retrieve results from processes"
                )
                # colors = [res.get(timeout=args.timeout) for res in colors]
                # colors = colors.get(timeout=args.timeout)
                # import pdb
                # pdb.set_trace()
                print("merging results")
                common_shape = next(
                    c.shape
                    for v in colors
                    for c in v.components()
                    if not isinstance(c, int)
                )
                color = rgb(
                    *[
                        np.concatenate(
                            [
                                c if type(c) != int else np.zeros(common_shape)
                                for c in comp
                            ]
                        )
                        for comp in zip(*[v.components() for v in colors])
                    ]
                )
                t2 = time.time()
                print(f"Took {t2-t1} seconds to merge results")
                # Q = vec3(x, y, 0)
                # color = do_raytrace(L, E, Q, scene, args.bounces)

                new_rgb = [
                    (255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8)
                    for c in color.components()
                ]
                # Image.fromarray(
                # Image.merge("RGB", new_rgb).save("fig.png")

                # screen_array = pygame.surfarray.pixels3d(screen)
                # print(screen_array.shape)
                # screen_array = new_rgb.T
                # del screen_array

                new_rgb = np.stack(new_rgb).T
            pygame.surfarray.blit_array(screen, new_rgb)
            display.blit(screen, (0, 0))

            pygame.display.update()
    t3 = time.time()
    print(f"Took {t3-t2} seconds to save results")
    pygame.quit()


if __name__ == "__main__":
    main(parser.parse_args())
