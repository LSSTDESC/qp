from mpi4py import MPI
import qp

# get the rank and size from MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# choose the chunk size for each iteration
chunk_size = 5

# the path to the file to iterate through
file_path = "./test.hdf5"

# iterate through the file
for start, end, ens_chunk in qp.iterator(
    file_path, chunk_size=chunk_size, rank=rank, parallel_size=size
):
    print(f"For rank: {rank}")
    print(f"Indices are: ({start}, {end})")
    print(ens_chunk)
