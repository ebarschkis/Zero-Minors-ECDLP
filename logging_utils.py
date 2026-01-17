import logging
import sys
from mpi4py import MPI


def get_logger(name: str = "apm") -> logging.Logger:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logger = logging.getLogger(f"{name}.r{rank}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [r%(rank)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    handler.addFilter(RankFilter())
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
