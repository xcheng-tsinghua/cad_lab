import math
import numpy as np


class OspPoint(object):
    """
    表示空间中的点，也可以当向量使用
    """
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

        # 给角点用的 id
        # self.id = None

    def __add__(self, other):
        assert isinstance(other, OspPoint), TypeError("add requires another OspPoint")
        return OspPoint(self.x + other.x,
                        self.y + other.y,
                        self.z + other.z)

    def __sub__(self, other):
        assert isinstance(other, OspPoint), TypeError("requires another OspPoint")
        return OspPoint(self.x - other.x,
                        self.y - other.y,
                        self.z - other.z)

    def __mul__(self, scalar):
        assert isinstance(scalar, (int, float)), TypeError("requires another OspPoint")
        return OspPoint(self.x * scalar,
                        self.y * scalar,
                        self.z * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        assert scalar != 0, ZeroDivisionError("division by zero")
        return OspPoint(self.x / scalar,
                        self.y / scalar,
                        self.z / scalar)

    def __neg__(self):
        return OspPoint(-self.x, -self.y, -self.z)

    # =========================
    # 向量运算
    # =========================

    def dot(self, other):
        """
        数量积（点积）
        :param other: OspPoint
        :return: float
        """
        assert isinstance(other, OspPoint), TypeError("dot product requires another OspPoint")

        dest = self.x * other.x + self.y * other.y + self.z * other.z
        return dest

    def cross(self, other):
        """
        向量积（叉积）
        :param other: OspPoint
        :return: OspPoint
        """
        assert isinstance(other, OspPoint), TypeError("cross product requires another OspPoint")

        cx = self.y * other.z - self.z * other.y
        cy = self.z * other.x - self.x * other.z
        cz = self.x * other.y - self.y * other.x

        return OspPoint(cx, cy, cz)

    def norm(self):
        """
        向量模长（L2）
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm2(self):
        """
        模长平方（避免开方，常用于比较）
        """
        return self.x**2 + self.y**2 + self.z**2

    def normalize(self):
        """
        单位化（返回新向量）
        """
        n = self.norm()
        assert n != 0, ValueError("cannot normalize zero-length vector")
        return self / n

    def angle_with(self, other):
        """
        计算与另一向量的夹角（弧度）
        """
        assert isinstance(other, OspPoint), TypeError("angle computation requires OspPoint")

        denom = self.norm() * other.norm()
        if denom == 0:
            raise ValueError("angle with zero-length vector")

        cos_theta = max(-1.0, min(1.0, self.dot(other) / denom))
        return math.acos(cos_theta)

    def is_parallel(self, other, tol=1e-8):
        """
        判断是否平行
        """
        return self.cross(other).norm() < tol

    def is_orthogonal(self, other, tol=1e-8):
        """
        判断是否正交
        """
        return abs(self.dot(other)) < tol

    # =========================
    # 点相关几何
    # =========================

    def distance_to(self, other):
        """
        点到点距离
        """
        return (self - other).norm()

    def project_onto(self, other):
        """
        投影到另一向量上
        """
        if other.norm2() == 0:
            raise ValueError("cannot project onto zero vector")
        return other * (self.dot(other) / other.norm2())

    # =========================
    # numpy / OCCT 友好接口
    # =========================

    def to_numpy(self):
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_numpy(cls, arr):
        arr = np.asarray(arr)
        assert arr.shape == (3,)
        return cls(arr[0].item(), arr[1].item(), arr[2].item())

    @classmethod
    def from_list(cls, coor_list):
        assert len(coor_list) == 3
        return cls(coor_list[0], coor_list[1], coor_list[2])

    def to_tuple(self):
        return self.x, self.y, self.z

    # =========================
    # 表示与比较
    # =========================

    def __repr__(self):
        return f'OspPoint({self.x:.6g}, {self.y:.6g}, {self.z:.6g})'

    def almost_equal(self, other, tol=1e-8):
        return (
            abs(self.x - other.x) < tol and
            abs(self.y - other.y) < tol and
            abs(self.z - other.z) < tol
        )


class OspCoordSystem(object):
    """
    表示空间中的坐标系
    """
    def __init__(self,
                 origin: OspPoint,
                 z_axis: OspPoint,
                 x_axis: OspPoint
                 ):

        self.origin = origin
        self.z_axis = z_axis
        self.x_axis = x_axis

    @classmethod
    def from_parsed_ofs(cls, parsed_ofs):
        """
        从解析特征列表获得的拓扑信息中的坐标系字典来初始化坐标系
        :param parsed_ofs:
        :return:
        """
        origin = OspPoint.from_list(parsed_ofs['origin'])
        z_axis = OspPoint.from_list(parsed_ofs['zAxis'])
        x_axis = OspPoint.from_list(parsed_ofs['xAxis'])

        return cls(origin, z_axis, x_axis)


