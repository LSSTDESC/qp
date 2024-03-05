from functools import reduce
from itertools import chain
from operator import add
import time

import numpy as np
import tqdm
from mpi4py import MPI

from pytdigest import TDigest


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# tdigest compression
COMPRESSION = 1000

# properties of lognormal distribution to be sampled
MEAN = 1
SIGMA = 1

# data volume
TOTAL_SIZE = 1_000_000
CHUNK_SIZE = TOTAL_SIZE // comm.size
N_BATCH = 10
BATCH_SIZE = CHUNK_SIZE // N_BATCH


def accumulate_digest_centroids(xs, compression=100):
    digest = TDigest.compute(xs, compression=compression)
    centroids = digest.get_centroids()

    return centroids


def main():
    rng = np.random.default_rng()

    start = time.time()
    centroids = []
    for i_batch in range(N_BATCH):
        xs = rng.lognormal(mean=MEAN, sigma=SIGMA, size=BATCH_SIZE)
        _centroids = accumulate_digest_centroids(xs, compression=COMPRESSION)
        centroids.append(_centroids)
    end = time.time()
    print(f"[MPI {rank}] processed {N_BATCH * BATCH_SIZE} objects in {end - start:.3g} seconds")

    centroids = comm.gather(centroids, root=0)

    if rank == 0:
        start = time.time()
        digests = (
            TDigest.of_centroids(centroid, compression=COMPRESSION)
            for centroid in chain.from_iterable(centroids)
        )
        td = reduce(add, digests)
        end = time.time()
        print(f"[MPI {rank}] reduced {N_BATCH * BATCH_SIZE * comm.size} objects in {end - start:.3g} seconds")

        # quantiles = [0., 0.25, 0.5, 0.75, 1.]
        # print(td.inverse_cdf(quantiles))
        print(f"true median: {np.exp(MEAN)}")
        print(f"approx median: {td.inverse_cdf(0.5)}")

if __name__ == "__main__":
    main()