"""
表示直线，
"""
import math
from functions.onshape.OspPoint import OspPoint
import numpy as np
from functions.onshape import macro, utils


def point_list_to_numpy(osp_point_list):
    np_list = []
    for c_osp in osp_point_list:
        np_list.append(c_osp.to_numpy())

    np_list = np.vstack(np_list)
    return np_list


class OspLine(object):
    def __init__(self, start_point, end_point):
        """
        直线需要空间中的始末点
        :param start_point:
        :param end_point:
        """
        self.start_point = start_point
        self.end_point = end_point
        self.sketch_plane = None

    def map_to_3d(self, sketch_plane):
        self.start_point = sketch_plane(self.start_point)
        self.end_point = sketch_plane(self.end_point)
        self.sketch_plane = sketch_plane

    def sample(self, num=50, include_end=True):
        """
        在该线段上均匀采样一系列连续点

        Parameters
        ----------
        num : int
            采样点数量（>=2）
        include_end : bool
            是否包含终点

        Returns
        -------
        List[OspPoint]
        """
        if num < 2:
            raise ValueError("num must be >= 2")

        direction = self.end_point - self.start_point

        # 参数 t 的取值
        if include_end:
            ts = [i / (num - 1) for i in range(num)]
        else:
            ts = [i / num for i in range(num)]

        points = []
        for t in ts:
            c_point = self.start_point + t * direction
            points.append(c_point)  # 保持 OspPoint 类型

        return point_list_to_numpy(points)


class OspCircle(object):
    """
    表示圆或者圆弧
    圆弧比整圆多两个端点
    z方向呈右手系旋转
    TODO: 添加圆弧的表达实现
    """
    def __init__(self, center_point, radius, normal, start_point, end_point):
        """

        :param center_point:
        :param radius:
        :param normal:
        """
        self.center_point = center_point
        self.radius = radius
        self.normal = normal
        self.sketch_plane = None

    def sample(self, num=50, include_end=True):
        """
        在圆上均匀采样一系列连续点（3D）

        Parameters
        ----------
        num : int
            采样点数量
        include_end : bool
            是否包含 2π 处的重复点

        Returns
        -------
        List[OspPoint]
        """
        if num < 3:
            raise ValueError("num must be >= 3")

        center = self.center_point
        r = self.radius

        if self.normal.norm == 0:
            raise ValueError("normal vector cannot be zero")

        n = self.normal.normalize()

        # ---------- 2. 构造平面内的 u 向量 ----------
        # 选一个不与 n 共线的参考向量
        if abs(n.x) < 0.9:
            a = OspPoint(1.0, 0.0, 0.0)
        else:
            a = OspPoint(0.0, 1.0, 0.0)

        # u = n × a
        u = n.cross(a).normalize()

        # ---------- 3. v = n × u ----------
        v = n.cross(u).normalize()

        # ---------- 4. 参数角 ----------
        if include_end:
            thetas = [2 * math.pi * i / (num - 1) for i in range(num)]
        else:
            thetas = [2 * math.pi * i / num for i in range(num)]

        points = []
        for theta in thetas:
            ct = math.cos(theta)
            st = math.sin(theta)

            dest = center + r * (ct * u + st * v)
            points.append(dest)

        return point_list_to_numpy(points)


class Ellipse(object):
    """
    表达椭圆
    """
    def __init__(self, origin, coor_x, coor_z, major_radius, minor_radius, start_point, end_point):
        """
        椭圆需要 中心位置、长轴长度、短轴长度、坐标系 X 方向为长轴方向，Z方向以右手系表达旋转方向
        椭圆弧比椭圆多两个端点
        """
        self.origin = origin
        self.coor_x = coor_x
        self.coor_z = coor_z
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.start_point = start_point
        self.end_point = end_point


class BSpline(object):
    """
    onshape 上除了 直线、圆、椭圆 其它曲线全用 BSpline 表示
    """
    def __init__(self,
                 control_points,  # point list
                 degree,  # int
                 dimension,  # int
                 is_periodic,  # bool
                 is_rational,  # bool
                 knots,  # float list
                 ):
        """
        椭圆需要长轴长度、短轴长度、坐标系 X 方向为长轴方向，Z方向以右手系表达旋转方向
        椭圆弧比椭圆多两个端点
        """
        self.control_points = control_points
        self.degree = degree
        self.dimension = dimension
        self.is_periodic = is_periodic
        self.is_rational = is_rational
        self.knots = knots




