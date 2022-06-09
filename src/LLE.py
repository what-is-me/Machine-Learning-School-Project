"""LLE降维"""
import numpy as np
from scipy.linalg import eigh, solve
from scipy.sparse import eye, csr_matrix
from sklearn.neighbors import NearestNeighbors


def DimensionalityReduction(
    data: np.ndarray, k: int = 2, m: int = 10, reg: float = 1e-3
) -> np.ndarray:
    """LLE降维

    Args:
        data (np.ndarray): 向量列表
        k (int): 降到k维. Defaults to 2.
        m (int): 近邻数(极大影响LLE性能). Defaults to 5.
        reg (float): 附加参数,用于防止某矩阵无法求逆. Defaults to 1e-3.

    Returns:
        np.ndarray: 处理后的向量列表
    """
    data = np.array(data)
    nbrs = NearestNeighbors(n_neighbors=m + 1).fit(data)
    nbrs_p = nbrs.kneighbors(return_distance=False)[
        :, 1:
    ]  # 每个点的m近邻（的位置）,因为低复杂度的k近邻实现较为复杂，这里直接调用现成的库了
    w = np.empty(nbrs_p.shape, dtype=data.dtype)
    v = np.ones(m, dtype=data.dtype)
    for i, j in enumerate(nbrs_p):
        C = data[j] - data[i]
        Zi = np.dot(C, C.T)
        Zi.flat[:: m + 1] += reg  # 防止Zi为奇异矩阵
        wi = solve(Zi, v, sym_pos=True)  # 求wi
        w[i, :] = wi / np.sum(wi)
    W = csr_matrix(
        (w.ravel(), nbrs_p.ravel(), np.arange(0, len(w) * m + 1, m)),
        shape=(len(w), len(w)),
    )  # 权重系数矩阵,以稀疏矩阵储存
    M = eye(*W.shape, format=W.format) - W
    M = (M.T * M).tocsr().toarray()  # 计算矩阵M=(I-W).T*(I-W)
    return eigh(M, eigvals=(1, k), overwrite_a=True)[1]  # M的前k个特征向量
