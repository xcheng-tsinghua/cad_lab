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
                 mid_point: OspPoint,  # 建模空间三维点，用于确定圆弧旋转方向
                 end_point: OspPoint,  # 建模空间三维点，闭合时为 None
                 topo_id: str
                 ):
        self.coord_sys = coord_sys
        self.radius = radius
        self.start_point = start_point
        self.mid_point = mid_point
        self.end_point = end_point
        self.id = topo_id

    def sample(self, n_samples: int = 50):
        """
        在圆 / 圆弧上均匀采样点（角度等分），用于可视化

        :param n_samples: 采样点数量
        :return: (n_samples, 3) 的 numpy.ndarray
        """
        C = self.coord_sys.origin          # 圆心
        X = self.coord_sys.x_axis.normalize()
        Z = self.coord_sys.z_axis.normalize()
        Y = Z.cross(X).normalize()      # 右手系

        R = self.radius

        # ========= 情况1：整圆 =========
        if self.start_point is None and self.end_point is None:
            theta_start = 0.0
            theta_end = 2 * math.pi

        # ========= 情况2：圆弧 =========
        else:
            v_start = (self.start_point - C)
            v_mid = (self.mid_point - C)
            v_end = (self.end_point - C)

            theta_start = math.atan2(v_start.dot(Y), v_start.dot(X))  # [-pi, pi]
            theta_mid = math.atan2(v_mid.dot(Y), v_mid.dot(X))
            theta_end = math.atan2(v_end.dot(Y), v_end.dot(X))

            # 统一到 [0, 2π)
            def wrap(theta):
                return theta + 2 * math.pi if theta < 0 else theta

            theta_start = wrap(theta_start)
            theta_mid = wrap(theta_mid)
            theta_end = wrap(theta_end)
            theta_start, theta_end = sorted((theta_start, theta_end))

            # 圆弧不横跨 2pi
            if not theta_start < theta_mid < theta_end:
                theta_start, theta_end = theta_end, theta_start
                theta_end += 2 * math.pi

        # ========= 采样 =========
        points = []
        for i in range(n_samples):
            t = theta_start + (theta_end - theta_start) * i / (n_samples - 1)
            p = C + (X * (R * math.cos(t))) + (Y * (R * math.sin(t)))
            points.append(p)

        return points


class OspEllipse(object):
    """
    表达椭圆
    """
    def __init__(self,
                 coord_sys: OspCoordSystem,  # x 轴为长轴方向，z轴为所在平面法向，该方向不代表圆弧旋转方向
                 major_radius: float,
                 minor_radius: float,
                 start_point: OspPoint,  # 建模空间三维点，闭合时为 None
                 mid_point: OspPoint,  # 建模空间三维点，用于确定椭圆旋转方向
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
        self.mid_point = mid_point
        self.end_point = end_point
        self.id = topo_id

    def sample(self, n_samples: int = 50):
        """
        在椭圆 / 椭圆弧上均匀采样点（按参数角度 θ 等分）

        :param n_samples: 采样点数量
        :return: (n_samples, 3) numpy.ndarray
        """
        C = self.coord_sys.origin
        X = self.coord_sys.x_axis.normalize()   # 长轴方向
        Z = self.coord_sys.z_axis.normalize()   # 平面法线
        Y = Z.cross(X).normalize()              # 短轴方向（自动）

        a = self.major_radius
        b = self.minor_radius

        # ========= 整椭圆 =========
        if self.start_point is None and self.end_point is None:
            ts = 0.0
            te = 2 * math.pi

        # ========= 椭圆弧 =========
        else:
            def param_angle(p):
                """
                椭圆参数角 θ（不是几何极角）
                """
                v = p - C
                x = v.dot(X)
                y = v.dot(Y)
                return math.atan2(y / b, x / a)

            ts = param_angle(self.start_point)
            tm = param_angle(self.mid_point)
            te = param_angle(self.end_point)

            # 统一到 [0, 2π)
            def wrap(theta):
                return theta + 2 * math.pi if theta < 0 else theta

            ts, tm, te = wrap(ts), wrap(tm), wrap(te)
            ts, te = sorted((ts, te))

            if not ts < tm < te:
                ts, te = te, ts
                te += 2 * math.pi

        # ========= 采样 =========
        pts = []
        for i in range(n_samples):
            t = ts + (te - ts) * i / (n_samples - 1)
            p_sample = C + X * (a * math.cos(t)) + Y * (b * math.sin(t))
            pts.append(p_sample)

        return pts


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


class OspFace(object):
    """
    一个草图可能存在多个区域，拉伸、旋转等都是选中区域进行操作
    """
    def __init__(self, edges: list, topo_id: str):
        self.id = topo_id
        self.edges = edges


