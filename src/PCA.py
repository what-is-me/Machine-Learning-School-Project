"""主成分分析降维"""
import numpy as np


def DimensionalityReduction(data: np.ndarray, k: int = 2) -> np.ndarray:
    """主成分分析法降维数据

    Args:
        arr ( np.ndarray): 向量列表
        k (int): 降至k维

    Returns:
        np.ndarray: 降维后的k维向量列表
    """
    ret = list(map(lambda x: np.array(x), data))
    m = sum(ret) / len(ret)
    ret = list(map(lambda x: x - m, ret))  # 标准化处理
    sigma = sum(np.matmul(np.mat(x).T, np.mat(x)) for x in ret) / len(ret)  # 协方差矩阵
    feature = sorted(
        zip(*map(lambda x: x.tolist(), np.linalg.eig(sigma))), key=lambda x: abs(x[0]), reverse=True
    )  # 特征值和特征向量，按特征值从大到小排列
    Mat = np.array([x[1] for x in feature[:k]])  # type: ignore
    # 取前k个特征向量构成投影矩阵
    return np.array([np.matmul(Mat, np.array(ret[i]).T) for i in range(len(ret))])
