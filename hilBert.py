import math


class Point:
    def __init__(self, x=0, y=0):
        self.x = x  # X坐标
        self.y = y  # Y坐标


class Hilbert:
    def __init__(self):
        # 初始化时计算 n=128, 64, 32, 16, 8, 4 的 Hilbert 代码到 XY 坐标映射
        self.hilbert_maps = {}
        for n in [64, 32, 16, 8, 4]:
            self.hilbert_maps[n] = self.precompute_hilbert_map(n)

    def precompute_hilbert_map(self, n):
        """预先计算给定 n 的 Hilbert 代码到 XY 坐标的映射"""
        hilbert_map = {}
        for d in range(n * n):  # 遍历所有可能的 Hilbert 代码
            pt = Point()
            self.d2xy(n, d, pt)
            hilbert_map[d] = (pt.x, pt.y)
        return hilbert_map

    def rot(self, n, pt, rx, ry):
        if ry == 0:
            if rx == 1:
                pt.x = n - 1 - pt.x
                pt.y = n - 1 - pt.y

            # Swap x and y
            pt.x, pt.y = pt.y, pt.x

    # Hilbert代码到XY坐标
    def d2xy(self, n, d, pt):
        pt.x, pt.y = 0, 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            self.rot(s, pt, rx, ry)
            pt.x += s * rx
            pt.y += s * ry
            t //= 4
            s *= 2

    # XY坐标到Hilbert代码转换
    def xy2d(self, n, pt):
        d = 0
        s = n // 2
        while s > 0:
            rx = 1 if (pt.x & s) > 0 else 0
            ry = 1 if (pt.y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            self.rot(s, pt, rx, ry)
            s //= 2
        return d

    def main(self):
        n = 4
        for i in range(n):
            for j in range(n):
                print(f"{self.xy2d(n, Point(j, i)):2}", end=" ")
            print()


if __name__ == "__main__":
    hilbert = Hilbert()
    # hilbert.main()
    map = hilbert.hilbert_maps
    print(len(map[8]))
    print(map[32][4][0])
    print(4099%4096)
    # print( 8192 // 4096)
    # for mapping in hilbert.hilbert_map.items():
    #     print(mapping)
