from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PCA
from Zipdata import Zipdata
import LLE
import Kmean
from random import randint as rand
import os
from tqdm import tqdm

os.system("mkdir image&cd image&mkdir LLE&mkdir PCA&mkdir colored&mkdir table")


def saveimg(pic: list | np.ndarray, path: str = "image/output") -> None:
    """将降维后的图片按维度显示

    Args:
        pic (list | np.ndarray): 图片
    """
    pic = np.array(pic)
    for i, p in zip(tqdm(range(len(pic)), desc="savepic"), pic):
        p = np.array([[complex(b).real for b in a] for a in p])
        maxn = p.max()
        minn = p.min()
        bd = maxn - minn
        plt.axis("off")
        plt.imshow(Image.fromarray((p - minn) / bd * 256), cmap="gray")
        plt.savefig(path + str(i))


# 数据读取
Data = gdal.Open("dc.tif").ReadAsArray()
size = 50
X, Y = 100, 100
data = [[row[X : X + size] for row in color[Y : Y + size]] for color in Data]
Data = [[[] for _ in range(size)] for _ in range(size)]
zipdata = Zipdata(size)
for color in data:
    for i in range(size):
        for j in range(size):
            Data[i][j].append(float(color[i][j]))
Data = np.array(zipdata.Zip(Data))  # 将二维矩阵展开成一维的


def DR(dr, Data: np.ndarray, N: int, k: int, Type: str):
    result = dr(Data, N)
    saveimg(zipdata.ToKPics(result), path=f"image/{Type}/img")
    color = [[rand(0, 255), rand(0, 255), rand(0, 255)] for _ in range(k)]  # 颜色随机
    # Kmean.SSE(Data, f"image/table/{Type}")
    pic = zipdata.Unzip(
        list(map(lambda x: color[x], Kmean.KMeans(result, k)[0]))
    )  # 将k-means分类后的结果替换成不同的颜色
    plt.imshow(np.array(pic))
    plt.savefig(f"image/colored/{Type}")  # 显示图片并保存
    print(f"{Type} over")


DR(PCA.DimensionalityReduction, Data, 50, 9, "PCA")
DR(LLE.DimensionalityReduction, Data, 10, 12, "LLE")
