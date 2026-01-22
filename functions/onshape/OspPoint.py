import math
import numpy as np


class OspPoint(object):
    """
    表示空间中的点，也可以当向量使用
    """
    __slots__ = ('x', 'y', 'z', 'is_2d')

    def __init__(self, x: float, y: float, z: float, is_2d: bool):
        self.x = x
        self.y = y
        self.z = z
        self.is_2d = is_2d
        assert (not self.is_2d) or (self.is_2d and self.z == 0)

    def __add__(self, other):
        assert isinstance(other, OspPoint), TypeError("add requires another OspPoint")
        assert self.is_2d == other.is_2d
        return OspPoint(self.x + other.x,
                        self.y + other.y,
                        self.z + other.z, self.is_2d)

    def __sub__(self, other):
        assert isinstance(other, OspPoint), TypeError("requires another OspPoint")
        assert self.is_2d == other.is_2d
        return OspPoint(self.x - other.x,
                        self.y - other.y,
                        self.z - other.z, self.is_2d)

    def __mul__(self, scalar):
        assert isinstance(scalar, (int, float)), TypeError("requires another OspPoint")
        return OspPoint(self.x * scalar,
                        self.y * scalar,
                        self.z * scalar, self.is_2d)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        assert scalar != 0, ZeroDivisionError("division by zero")
        return OspPoint(self.x / scalar,
                        self.y / scalar,
                        self.z / scalar, self.is_2d)

    def __neg__(self):
        return OspPoint(-self.x, -self.y, -self.z, self.is_2d)

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
        assert self.is_2d == other.is_2d

        dest = self.x * other.x + self.y * other.y + self.z * other.z
        return dest

    def cross(self, other):
        """
        向量积（叉积）
        :param other: OspPoint
        :return: OspPoint
        """
        assert isinstance(other, OspPoint), TypeError("cross product requires another OspPoint")
        assert self.is_2d == other.is_2d

        cx = self.y * other.z - self.z * other.y
        cy = self.z * other.x - self.x * other.z
        cz = self.x * other.y - self.y * other.x

        if self.is_2d:
            return cz
        else:
            return OspPoint(cx, cy, cz, self.is_2d)

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
        assert self.is_2d == other.is_2d

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

    @staticmethod
    def from_numpy(arr):
        arr = np.asarray(arr)
        if arr.shape == (3,):
            return OspPoint(arr[0].item(), arr[1].item(), arr[2].item(), False)

        elif arr.shape == (2,):
            return OspPoint(arr[0].item(), arr[1].item(), 0, True)

        else:
            raise ValueError("cannot convert this ndarray")

    def to_tuple(self):
        if self.is_2d:
            return self.x, self.y

        else:
            return self.x, self.y, self.z

    # =========================
    # 表示与比较
    # =========================

    def __repr__(self):
        return f'OspPoint({self.x:.6g}, {self.y:.6g}, {self.z:.6g})'

    def almost_equal(self, other, tol=1e-8):
        assert self.is_2d == other.is_2d

        return (
            abs(self.x - other.x) < tol and
            abs(self.y - other.y) < tol and
            abs(self.z - other.z) < tol
        )
