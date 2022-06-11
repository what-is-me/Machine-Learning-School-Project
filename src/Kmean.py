from typing import Union
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def dis(x: list, y: list) -> int:
    return sum((d1 - d2) ** 2 for d1, d2 in zip(x, y))


def distance(p1, p3):
    return min(dis(p1, p2) for p2 in p3)


def _kmeans(data: np.ndarray, k: int):
    print(k)
    center = [data[0]]
    center.extend(
        data[np.argmax(np.array([distance(point, center) for point in data]))] for _ in range(k - 1)
    )
    points = [[] for _ in range(k)]
    x = 0.002
    i, j = 0, 0
    while x > 0.001:
        points = [[] for _ in range(k)]
        for point in data:
            points[np.argmin(np.array([dis(point, cp) for cp in center]))].append(point)
        center = list(map(lambda lst: np.array(lst).T.mean(axis=1), points))
        i, j = j, 0
        for l in range(k):
            for pot in points[l]:
                j += float(sum((x - y) ** 2 for x, y in zip(center[l], pot)))
        x = abs(i - j)
    return np.array(points), np.array(center)


def SSE(data: np.ndarray, path: str) -> None:
    K = list(range(1, 50))
    sse_result = []
    for k in K:
        p, c = _kmeans(data, k)
        s = 0
        for l in range(k):
            for pot in p[l]:
                s += float(sum((x - y) ** 2 for x, y in zip(c[l], pot)))
        sse_result.append(s)
    plt.plot(K, sse_result, "gx-")
    plt.xlabel("k")
    plt.ylabel(u"平均畸变程度")
    plt.title(u"肘部法则确定最佳的K值")
    plt.savefig(path)


def KMeans(data: np.ndarray, k: int, times: int = 10):
    center = [data[0]]
    center.extend(
        data[np.argmax(np.array([distance(point, center) for point in data]))] for _ in range(k - 1)
    )
    color = np.empty((len(data),), dtype=int)
    for _ in tqdm(range(times), desc="K-means"):
        for i in range(len(data)):
            color[i] = np.argmin([dis(data[i], v) for v in center])
        center = np.zeros((k, len(data[0])), dtype=np.float64)
        num = np.zeros(k, dtype=int)
        for i in range(len(data)):
            num[color[i]] += 1
            center[color[i]] += data[i]
        for i in range(k):
            if num[i] != 0:
                center[i] /= num[i]
    return color, center
