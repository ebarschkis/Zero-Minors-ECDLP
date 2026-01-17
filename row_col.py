from dataclasses import dataclass, field
from typing import List


@dataclass
class RowCol:
    dimension: int = 0
    rows: List[int] = field(default_factory=list)
    cols: List[int] = field(default_factory=list)

    def __init__(self, dimension: int = 0):
        self.dimension = dimension
        self.rows = [0] * dimension
        self.cols = [0] * dimension

    def init(self) -> None:
        for i in range(self.dimension):
            self.rows[i] = i
            self.cols[i] = i

    def print(self, msg: str = "") -> None:
        if self.dimension == 0:
            return
        if msg:
            print(msg)
        print("\t".join(str(x) for x in self.rows))
        print("\t".join(str(x) for x in self.cols))

    def copy(self) -> "RowCol":
        rc = RowCol(self.dimension)
        rc.rows = self.rows[:]
        rc.cols = self.cols[:]
        return rc
