#!/usr/bin/env python3
import time
import os
import multiprocessing
import copy
import argparse

import numpy as np
from PIL import Image
import pygame
from pygame.locals import (
    KEYDOWN,
    K_ESCAPE,
    K_UP,
    K_DOWN,
    K_d,
    K_s,
    K_a,
    K_w,
    K_e,
    K_q,
    QUIT,
    K_n,
    K_p,
    K_b,
    KMOD_LSHIFT,
    KMOD_LCTRL
)


from core_raytrace import (
    vec3,
    rgb,
    next_highest_divisor,
    do_raytrace_v2,
    load_and_parse_scene_from_file,
)


parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=800)
parser.add_argument("--height", type=int, default=640)
parser.add_argument("--bounces", type=int, default=3)
parser.add_argument("--refractions", type=int, default=3)
parser.add_argument("--timeout", type=int, default=30)
parser.add_argument("--processes", type=int, default=None)
parser.add_argument("--scenefile", type=str, default="scene.json")


def main(args):
    pygame.init()
    SIZE = (w, h) = (args.width, args.height)  # Screen size
    display = pygame.display.set_mode(SIZE)

    screen = pygame.Surface(SIZE)

    L = vec3(5, 5.0, -5.0)  # Point light position
    E = vec3(0.0, 0.35, -1.0)  # Eye position
    S = vec3(0.0, 0.15, 0.0)  # Viewport center position
    S_DIST = 1
    S_ROT_LR = 0
    S_ROT_UD = 0
    S_SIZE = (2, 2 * h / w)  # Viewport size in world coordinates

    # with eye at vec3(0.0, 0.35, -1.0),
    # x is left right,
    # y is up down
    # z is close far.
    scene = load_and_parse_scene_from_file(args.scenefile)

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    # fmt:off

    def compute_viewport(S):
        return (-S_SIZE[0] / 2, S_SIZE[1] / 2, S_SIZE[0] / 2, - S_SIZE[1] / 2, S_ROT_LR, S_ROT_UD)
    # fmt:on
    # x = np.tile(np.linspace(S[0], S[2], w), h)
    # y = np.repeat(np.linspace(S[1], S[3], h), w)
    # print(x.shape, y.shape)

    # N = 4
    oldN = os.cpu_count() if args.processes is None else args.processes
    N = next_highest_divisor(h, oldN)
    if N != oldN:
        print(f"rounding up number of processes to be a divisor of {h}. {h} % {N} == {h%N}")

    first_execution = True
    i = 0
    save_image = False
    with multiprocessing.Pool(processes=min(N, os.cpu_count() - 1)) as pool:
        while True:
            invalidated = False
            for e in pygame.event.get():
                if (e.type == KEYDOWN and e.key == K_ESCAPE) or e.type == QUIT:
                    return
                if e.type == KEYDOWN:
                    delta = 0.1
                    x = 0
                    y = 0
                    z = 0
                    lx = 0
                    ly = 0
                    lz = 0
                    LR = 0
                    UD = 0
                    bounce_delta = 0
                    if e.key == K_w:
                        if e.mod & KMOD_LSHIFT:
                            ly += 1
                        else:
                            y += 1
                    elif e.key == K_s:
                        if e.mod & KMOD_LSHIFT:
                            ly -= 1
                        else:
                            y -= 1
                    elif e.key == K_a:
                        if e.mod & KMOD_LSHIFT:
                            lx -= 1
                        else:
                            x -= 1
                    elif e.key == K_d:
                        if e.mod & KMOD_LSHIFT:
                            lx += 1
                        else:
                            x += 1
                    elif e.key == K_UP:
                        if e.mod & KMOD_LSHIFT:
                            lz += 1
                        elif e.mod & KMOD_LCTRL:
                            S_DIST += 1
                        else:
                            z += 1
                    elif e.key == K_DOWN:
                        if e.mod & KMOD_LSHIFT:
                            lz -= 1
                        elif e.mod & KMOD_LCTRL:
                            S_DIST -= 1
                        else:
                            z -= 1
                    elif e.key == K_b:
                        bounce_delta += 1
                    elif e.key == K_n:
                        bounce_delta -= 1
                    elif e.key == K_e:
                        if e.mod & KMOD_LSHIFT:
                            LR += 1
                        else:
                            UD += 1
                    elif e.key == K_q:
                        if e.mod & KMOD_LSHIFT:
                            LR -= 1
                        else:
                            UD -= 1
                    elif e.key == K_p:
                        print("pressed p")
                        save_image = True

                    if any(_ != 0 for _ in [x, y, z, lx, ly, lz, UD, LR, bounce_delta]):
                        invalidated = True

                        S.x += x * delta
                        S.y += z * delta
                        S.z += y * delta
                        E.x += x * delta
                        E.y += z * delta
                        E.z += y * delta
                        L.x += lx * delta
                        L.y += lz * delta
                        L.z += ly * delta
                        S_ROT_LR += LR * delta
                        S_ROT_UD += UD * delta
                        args.bounces += bounce_delta
                    # TODO: figure out how to rotate camera + eye positions.
            if invalidated or first_execution:
                print(L.components())
                print(E.components())
                print(S.components())
                print(S_ROT_UD, S_ROT_LR)
                first_execution = False
                t0 = time.time()
                print(f"starting pool execution on {N} processes")

                print("sending starmap order")
                colors = pool.starmap(
                    do_raytrace_v2,
                    [
                        copy.deepcopy(
                            tuple(
                                [
                                    L,
                                    E,
                                    S,
                                    compute_viewport(S),
                                    w,
                                    h,
                                    scene,
                                    i,
                                    N,
                                    args.bounces,
                                    args.refractions
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
                    c.shape for v in colors for c in v.components() if not isinstance(c, int)
                )
                color = rgb(
                    *[
                        np.concatenate(
                            [c if type(c) != int else np.zeros(common_shape) for c in comp]
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
            if save_image:
                print("saving image")
                image_rgb = [
                    Image.fromarray(c, "L")
                    for c in new_rgb.T
                ]
                Image.merge("RGB", image_rgb).save(f"fig{i}.png")
                i += 1
                save_image = False
            pygame.surfarray.blit_array(screen, new_rgb)
            display.blit(screen, (0, 0))

            pygame.display.update()
    t3 = time.time()
    print(f"Took {t3-t2} seconds to save results")
    pygame.quit()


if __name__ == "__main__":
    main(parser.parse_args())
