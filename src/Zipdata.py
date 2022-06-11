from numpy import ndarray, array


class Zipdata:
    def __init__(self, n):
        self.n = n

    def Zip(self, Data: ndarray) -> ndarray:
        """二维数组展成一维数组"""
        return array([i for item in Data for i in item])

    def ToKPics(self, Data: ndarray) -> ndarray:
        """一维数组转换成k个二维数组"""
        N = len(Data[0])
        return array(
            [
                [[Data[j * self.n + i][p] for i in range(self.n)] for j in range(self.n)]
                for p in range(N)
            ]
        )

    def Unzip(self, Data: ndarray) -> ndarray:
        """一维数组转回二维数组"""
        return array([[Data[j * self.n + i] for i in range(self.n)] for j in range(self.n)])
