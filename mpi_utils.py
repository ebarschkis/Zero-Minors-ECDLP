from mpi4py import MPI


def mpi_get_mpi_rank_size_name():
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    node_name = MPI.Get_processor_name()
    return processor_id, total, node_name


MPI_get_MPI_rank_size_name = mpi_get_mpi_rank_size_name
