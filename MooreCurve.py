import math


class MooreCurve:
    def __init__(self):
        self.mooreCurveMaps = {}
        # 预计算常见尺寸的映射表
        for side_length in [64, 32, 16, 8, 4]:
            self.mooreCurveMaps[side_length] = self.precompute_moore_curve_map(side_length)

    def precompute_moore_curve_map(self, side_length):
        """预计算指定边长的摩尔曲线映射表"""
        # 计算阶数（根据边长推导）
        n = int(math.log2(side_length))

        # 生成坐标序列
        points = self.moore_curve_order_to_coords(n)

        # 计算坐标范围
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)

        curve_map = {}
        # 填充遍历顺序
        for step, (x, y) in enumerate(points):
            curve_map[step + 1] = (y - min_y, x - min_x)

        return curve_map

    def generate_moore_curve_string(self, n):
        """生成n阶L-system字符串"""
        axiom = 'LFL+F+LFL'
        rules = {
            'L': '-RF+LFL+FR-',
            'R': '+LF-RFR-FL+',
        }
        current = axiom
        for _ in range(n - 1):  # 替换次数为n-1次
            new_str = []
            for char in current:
                new_str.append(rules.get(char, char))  # 应用规则或保留原字符
            current = ''.join(new_str)
        return current

    def moore_curve_order_to_coords(self, n):
        """解析L-system字符串生成坐标序列"""
        string = self.generate_moore_curve_string(n)

        # 初始化状态
        x, y = 0, 0
        direction = 0  # 0:西 1:北 2:东 3:南
        points = [(x, y)]  # 包含起点

        # 解析指令
        for char in string:
            if char == 'F':
                # 根据当前方向移动
                if direction == 0:
                    x += 1
                elif direction == 1:
                    y += 1
                elif direction == 2:
                    x -= 1
                elif direction == 3:
                    y -= 1
                points.append((x, y))
            elif char == '+':
                direction = (direction - 1) % 4  # 右转
            elif char == '-':
                direction = (direction + 1) % 4  # 左转

        return points

    @staticmethod
    def coords_to_2d_array(points):
        """将坐标序列转换为二维数组（辅助函数）"""
        if not points:
            return []

        # 计算坐标范围
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)

        # 初始化数组
        rows = max_y - min_y + 1
        cols = max_x - min_x + 1
        grid = [[0 for _ in range(cols)] for _ in range(rows)]

        # 填充遍历顺序
        for step, (x, y) in enumerate(points):
            grid[y - min_y][x - min_x] = step + 1

        return grid


# 使用示例
if __name__ == "__main__":
    curve = MooreCurve()

    # 获取边长4的映射表
    map_4x4 = curve.mooreCurveMaps[4]
    print("4x4摩尔曲线前5个坐标点:")
    for order in range(1, 17):
        print(f"顺序 {order}: {map_4x4[order]}")

    # 打印二维数组形式
    points = curve.moore_curve_order_to_coords(2)
    grid = MooreCurve.coords_to_2d_array(points)
    print("\n二维数组表示:")
    for row in grid:
        print(row)
