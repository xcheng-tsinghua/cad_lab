"""
表示直线，
"""
import math
from functions.onshape.OspGeomBase import OspPoint, OspCoordSystem
import numpy as np
from scipy.interpolate import BSpline


class OspLine(object):
    def __init__(self,
                 start_point: OspPoint,
                 end_point: OspPoint,
                 topo_id: str
                 ):
        """
        直线需要空间中的始末点
        :param start_point:
        :param end_point:
        :param topo_id:
        """
        self.start_point = start_point
        self.end_point = end_point
        self.id = topo_id

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

        return points


class OspCircle(object):
    """
    表示圆或者圆弧
    圆弧比整圆多两个端点
    """
    def __init__(self,
                 coord_sys: OspCoordSystem,
                 radius: float,
                 start_point: OspPoint,  # 建模空间三维点，闭合时为 None
                 end_point: OspPoint,  # 建模空间三维点，闭合时为 None
                 topo_id: str
                 ):
        """

        :param coord_sys: 局部坐标系，原点为圆心位置，Z方向呈右手系旋转方向
        :param radius: 半径
        :param start_point: 起始点
        :param end_point: 终止点
        """
        self.coord_sys = coord_sys
        self.radius = radius
        self.start_point = start_point
        self.end_point = end_point
        self.id = topo_id

    def sample(self, n_samples: int = 50):
        """
        在圆 / 圆弧上均匀采样点（角度等分），用于可视化

        :param n_samples: 采样点数量
        :return: (n_samples, 3) 的 numpy.ndarray
        """

        origin = self.coord_sys.origin
        x_axis = self.coord_sys.x_axis.normalize()
        z_axis = self.coord_sys.z_axis.normalize()

        # 构造 y 轴，保证右手系
        y_axis = z_axis.cross(x_axis).normalize()

        def point_to_angle(p):
            """
            将空间点映射为局部坐标系下的极角
            """
            v = p - origin
            x = v.dot(x_axis)
            y = v.dot(y_axis)
            return math.atan2(y, x)

        # 整圆
        if self.start_point is None and self.end_point is None:
            angles = np.linspace(
                0.0,
                2.0 * math.pi,
                n_samples,
                endpoint=False
            )
        else:
            if self.start_point is None or self.end_point is None:
                raise ValueError("圆弧必须同时提供 start_point 和 end_point")

            theta_start = point_to_angle(self.start_point)
            theta_end = point_to_angle(self.end_point)

            # 保证沿 z_axis 定义的右手方向
            if theta_end <= theta_start:
                theta_end += 2.0 * math.pi

            angles = np.linspace(theta_start, theta_end, n_samples)

        points = []

        for i, theta in enumerate(angles):
            p_sample = origin + x_axis * (self.radius * math.cos(theta)) + y_axis * (self.radius * math.sin(theta))

            points.append(p_sample)

        return points


class OspEllipse(object):
    """
    表达椭圆
    """
    def __init__(self,
                 coord_sys: OspCoordSystem,
                 major_radius: float,
                 minor_radius: float,
                 start_point: OspPoint,  # 建模空间三维点，闭合时为 None
                 end_point: OspPoint,  # 建模空间三维点，闭合时为 None
                 topo_id: str
                 ):
        """
        椭圆需要 中心位置、半长轴、半短轴、坐标系 X 方向为长轴方向，Z方向以右手系表达旋转方向
        椭圆弧比椭圆多两个端点
        """
        self.coord_sys = coord_sys
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.start_point = start_point
        self.end_point = end_point
        self.id = topo_id

    def sample(self, n_samples: int = 50):
        """
        在椭圆 / 椭圆弧上均匀采样点（按参数角度 θ 等分）

        :param n_samples: 采样点数量
        :return: (n_samples, 3) numpy.ndarray
        """

        origin = self.coord_sys.origin
        x_axis = self.coord_sys.x_axis.normalize()
        z_axis = self.coord_sys.z_axis.normalize()

        # 保证右手系
        y_axis = z_axis.cross(x_axis).normalize()

        a = self.major_radius
        b = self.minor_radius

        def point_to_angle(p):
            """
            将空间点映射为椭圆参数角 θ
            """
            v = p - origin
            x = v.dot(x_axis) / a
            y = v.dot(y_axis) / b
            return math.atan2(y, x)

        # 整椭圆
        if self.start_point is None and self.end_point is None:
            angles = np.linspace(
                0.0,
                2.0 * math.pi,
                n_samples,
                endpoint=False
            )
        else:
            if self.start_point is None or self.end_point is None:
                raise ValueError("椭圆弧必须同时提供 start_point 和 end_point")

            theta_start = point_to_angle(self.start_point)
            theta_end = point_to_angle(self.end_point)

            # 保证沿 z_axis 右手方向
            if theta_end <= theta_start:
                theta_end += 2.0 * math.pi

            angles = np.linspace(theta_start, theta_end, n_samples)

        points = []

        for i, theta in enumerate(angles):
            p_sample = origin + x_axis * (a * math.cos(theta)) + y_axis * (b * math.sin(theta))
            points.append(p_sample)

        return points


class OspBSpline(object):
    """
    onshape 上除了 直线、圆、椭圆 其它曲线全用 BSpline 表示
    """
    def __init__(self,
                 control_points: list[OspPoint],
                 degree: int,
                 dimension: int,
                 is_periodic: bool,
                 is_rational: bool,
                 knots: list[float],
                 weights: list[float],  # 非有理时为 None
                 start_point: OspPoint,  # 闭合时为 None
                 end_point: OspPoint,  # 闭合时为 None
                 topo_id: str,
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
        self.weights = weights
        self.start_point = start_point
        self.end_point = end_point
        self.id = topo_id

    def sample(self, n_samples: int = 50):
        """
        在 B-Spline / NURBS 曲线上按参数均匀采样点（用于可视化）

        :return: (n_samples, 3) numpy.ndarray
        """

        p = self.degree
        knots = np.asarray(self.knots, dtype=float)

        ctrl = np.array(
            [[p.x, p.y, p.z] for p in self.control_points],
            dtype=float
        )

        n = len(ctrl) - 1

        # 参数区间（标准 B-Spline）
        u_start = knots[p]
        u_end = knots[n + 1]
        us = np.linspace(u_start, u_end, n_samples)

        # ---------- 非有理 B-Spline ----------
        if not self.is_rational:
            # SciPy: 一个 BSpline 就是一个向量值函数
            spline = BSpline(knots, ctrl, p, extrapolate=False)
            points = spline(us)

        # ---------- 有理 B-Spline（NURBS） ----------
        else:
            weights = np.asarray(self.weights, dtype=float)

            # 构造齐次坐标
            ctrl_h = np.hstack([
                ctrl * weights[:, None],
                weights[:, None]
            ])

            spline_h = BSpline(knots, ctrl_h, p, extrapolate=False)
            vals = spline_h(us)

            # 齐次 -> 三维
            points = vals[:, :3] / vals[:, 3:4]

        points_osp = []
        for point_np in points:
            p_osp = OspPoint.from_numpy(point_np)
            points_osp.append(p_osp)

        return points_osp


class OspRegion(object):
    """
    一个草图可能存在多个区域，拉伸、旋转等都是选中区域进行操作
    """
    def __init__(self, primitive_list: list, topo_id: str):
        self.id = topo_id
        self.primitive_list = primitive_list


