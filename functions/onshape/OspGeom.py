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


def proj_to_plane(point, point_plane, normal_plane):
    """
    将空间点 point 正交投影到由 plane_point + plane_normal 定义的平面上

    Parameters
    ----------
    point : (3,) array-like
        待投影点 P
    point_plane : (3,) array-like
        平面上一点 P0
    normal_plane : (3,) array-like
        平面法向量 n（不要求单位化）

    Returns
    -------
    proj_point : (3,) ndarray
        投影点的三维坐标
    """
    # 法线归一化
    normal_plane = normal_plane.normalize()

    # 平面上一点到指定点的向量
    plane_to_point = point - point_plane

    # 计算该向量到法线的投影长度
    norm_proj_len = plane_to_point.dot(normal_plane)

    # 反向倒推回平面上的该点
    proj_point = point - norm_proj_len * normal_plane

    return proj_point


class OspLine(object):
    def __init__(self, start_point, end_point, is_2d):
        """

        :param start_point:
        :param end_point:
        :param is_2d:
        """
        self.start_point = start_point
        self.end_point = end_point
        self.is_2d = is_2d
        self.sketch_plane = None

        self.check()

    def map_to_3d(self, sketch_plane):
        self.start_point = sketch_plane(self.start_point)
        self.end_point = sketch_plane(self.end_point)
        self.is_2d = False
        self.sketch_plane = sketch_plane

        self.check()

    def check(self):
        # 直线本身的状态必须和始末点相同
        assert self.is_2d == self.start_point.is_2d == self.end_point.is_2d

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
        assert not self.is_2d

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
    def __init__(self, center_point, radius, normal, start_point, end_point, is_2d):
        """

        :param center_point:
        :param radius:
        :param normal:
        :param is_2d:
        """
        self.center_point = center_point
        self.radius = radius
        self.normal = normal
        self.is_2d = is_2d
        self.sketch_plane = None

        self.check()

    def map_to_3d(self, sketch_plane):
        self.center_point = sketch_plane(self.center_point)
        self.radius = sketch_plane(self.radius)
        self.normal = sketch_plane.normal
        self.is_2d = False
        self.sketch_plane = sketch_plane

        self.check()

    def check(self):
        # 圆心的点状态必须和该圆相同，且2d 圆的 normal 必须为 None，3d圆的法线不能为None
        assert self.is_2d == self.center_point.is_2d
        if self.is_2d:
            assert self.normal is None
        else:
            assert self.normal is not None

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
        assert not self.is_2d

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
            a = OspPoint(1.0, 0.0, 0.0, False)
        else:
            a = OspPoint(0.0, 1.0, 0.0, False)

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


class SketchPlane(object):
    def __init__(self, origin, normal, sketch_x, unit):
        """

        :param res_msg_val_item_ofs: 通过 api 解析出的 json 字典
        """
        self.origin = origin
        self.normal = normal
        self.sketch_x = sketch_x
        self.unit = unit

        # 由 onshape 定义，草图局部坐标系 y 方向等于 normal cross x
        self.sketch_y = self.normal.cross(self.sketch_x)

    @classmethod
    def from_ofs(cls, res_msg_val_item_ofs):
        """
        从 onshape 解析出的 feature_list 构造平面
        :param res_msg_val_item_ofs:
        :return:
        """
        val2nd_list_ofs = res_msg_val_item_ofs['message']['value']

        origin = None
        normal = None
        sketch_x = None
        units = []

        for val2nd_item_ofs in val2nd_list_ofs:
            val_type = val2nd_item_ofs['message']['key']['message']['value']
            val4th_ofs_list = val2nd_item_ofs['message']['value']['message']['value']
            if val_type == 'normal':
                normal = cls.extract_4th_ofs_point(val4th_ofs_list, False)

            elif val_type == 'origin':
                c_orign, units = cls.extract_4th_ofs_point(val4th_ofs_list, True)
                origin = c_orign
                units.extend(units)

            elif val_type == 'x':
                sketch_x = cls.extract_4th_ofs_point(val4th_ofs_list, False)

            elif val_type == 'surfaceType':
                assert val4th_ofs_list == 'PLANE'

            else:
                raise TypeError(f'this sketch plane is not plane: {val4th_ofs_list}')

        assert normal is not None and origin is not None and sketch_x is not None and units

        # 测试将局部坐标系原点永远与世界坐标系原点到平面的投影点重合
        origin = proj_to_plane(OspPoint(0, 0, 0, False), origin, normal)

        assert len(set(units)) == 1
        units = units[0]

        return cls(origin, normal, sketch_x, units)

    @staticmethod
    def extract_4th_ofs_point(val4th_ofs_list, with_units):
        """
        提取最深的三维点或向量
        :param val4th_ofs_list:
        :param with_units: 是否包含单位，法线和X方向都不包含单位，原点包含单位
        :return:
        """
        all_vals = []
        all_units = []

        for val4th_item_ofs in val4th_ofs_list:
            c_val = val4th_item_ofs['message']['value']
            all_vals.append(c_val)

            if with_units:
                c_unit_key = val4th_item_ofs['message']['unitToPower'][0]['key']
                c_unit_val = val4th_item_ofs['message']['unitToPower'][0]['value']

                c_unit = (c_unit_key, float(c_unit_val))
                all_units.append(c_unit)

        assert len(all_vals) == 3
        res = OspPoint(all_vals[0], all_vals[1], all_vals[2], False)

        if with_units:
            return res, all_units

        else:
            return res

    def __call__(self, other):
        """
        将该草图基准面上的二维点转化为三维空间中的点
        :param other:
        :return:
        """
        # onshape 默认英尺，导出时是米，因此换算
        mul_unit = utils.get_unit_trans_coff(self.unit, macro.GLOBAL_UNIT)

        if isinstance(other, OspPoint):

            assert other.is_2d

            point_3d = self.origin + other.x * self.sketch_x + other.y * self.sketch_y
            point_3d = mul_unit * point_3d
            assert not point_3d.is_2d
            return point_3d

        elif isinstance(other, (int, float)):
            return mul_unit * other

        else:
            raise NotImplementedError


