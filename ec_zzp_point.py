class EC_ZZp_Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.z = 1

    def printPoint(self, msg: str) -> None:
        print(f"{msg} :: [{self.x} : {self.y} : {self.z}]")


EC_ZZp_Point = EC_ZZp_Point
