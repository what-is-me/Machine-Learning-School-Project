from numpy import ndarray


class Zipdata:
    def __init__(self, n):
        self.n = n

    def Zip(self, Data: list[list[list]] | ndarray) -> list[list]:
        """二维数组展成一维数组"""
        return [i for item in Data for i in item]

    def ToKPics(self, Data: list[list] | ndarray) -> list[list[list]]:
        """一维数组转换成k个二维数组"""
        N = len(Data[0])
        return [
            [[Data[j * self.n + i][p] for i in range(self.n)] for j in range(self.n)]
            for p in range(N)
        ]

    def Unzip(self, Data: list[list] | ndarray) -> list[list[list]]:
        """一维数组转回二维数组"""
        return [[Data[j * self.n + i] for i in range(self.n)] for j in range(self.n)]
