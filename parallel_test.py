import multiprocessing
import numpy as np

def compute(i, batch):
    print(i, batch)
    s = 0
    for k in batch:
        for j in range(k):
            s += k*j*i
    return i * s

N = multiprocessing.cpu_count()

with multiprocessing.Pool(processes=N) as pool:
    Qs = np.split(np.arange(0,6000), N)
    results = pool.starmap_async(compute, [(i, sub) for i, sub in enumerate(Qs)])
    results = results.get(timeout=10)
    print(results)