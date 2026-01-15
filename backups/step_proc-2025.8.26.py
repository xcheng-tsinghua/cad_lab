# open cascade
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_ShapeEnum, TopAbs_REVERSED
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopoDS import topods
from OCC.Core.Precision import precision
from OCC.Core.GProp import GProp_PGProps, GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.Geom import Geom_ConicalSurface, Geom_Plane, Geom_CylindricalSurface, Geom_Curve, Geom_SphericalSurface
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepTools import breptools
# others
import os
import open3d as o3d
import pymeshlab
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
import multiprocessing
import itertools
from collections import Counter

# self
from functions import mesh_proc
from utils import utils


class Point3DForDataSet(gp_Pnt):
    """
    使用类处理点的额外属性
    """
    def __init__(self, pnt_loc: gp_Pnt, aligned_face: TopoDS_Face, aligned_shape_area, edges_useful):
        super().__init__(pnt_loc.XYZ())

        self.edges_useful = edges_useful

        self.aligned_face = aligned_face
        self.aligned_shape_area = aligned_shape_area

        self.pmt = -1
        self.mad = gp_Vec(0.0, 0.0, -1.0)
        self.is_edge_nearby = 0
        self.edge_nearby_threshold = 0.0

        self.pmt_mad_cal()
        self.edge_nearby_cal()

    def pmt_mad_cal(self):
        # 计算所在基元类型及主要方向
        aligned_surface = BRep_Tool.Surface(self.aligned_face)
        type_name = face_type(self.aligned_face)

        if type_name == 'cone':
            self.pmt = 1
            self.mad = Geom_ConicalSurface.DownCast(aligned_surface).Axis().Direction().XYZ()
            self.mad_rectify()  # 校正方向，使方向唯一

        elif type_name == 'cylinder':
            self.pmt = 2
            self.mad = Geom_CylindricalSurface.DownCast(aligned_surface).Axis().Direction().XYZ()
            self.mad_rectify()  # 校正方向，使方向唯一

        elif type_name == 'plane':
            self.pmt = 3
            self.mad = Geom_Plane.DownCast(aligned_surface).Axis().Direction().XYZ()
            self.mad_rectify()  # 校正方向，使方向唯一

        # elif type_name == 'Geom_SphericalSurface':
        #     self.pmt = 4
        #     self.mad = gp_Vec(0.0, 0.0, -1.0)

        else:
            # 其它不好定义的类型，欧拉角统一定义为 (0, 0, -1), 这批弄完后统一改
            self.pmt = 0

    def mad_rectify(self):
        """
        保证欧拉角方向唯一
        """
        ax_x = self.mad.X()
        ax_y = self.mad.Y()
        ax_z = self.mad.Z()

        zero_lim = precision.Confusion()
        if ax_z < -zero_lim:  # z < 0
            self.mad *= -1.0
        elif abs(ax_z) <= zero_lim and ax_y < -zero_lim:  # z为零, y为负数
            self.mad *= -1.0
        elif abs(ax_z) <= zero_lim and abs(ax_y) <= zero_lim and ax_x < -zero_lim:  # z为零, y为零, x为负数
            self.mad *= -1.0
        else:
            raise ValueError('error main axis direction length')

    # 计算边缘邻近阈值
    def nearby_threshold_cal(self):
        rsphere = np.sqrt(self.aligned_shape_area / (4.0 * np.pi))
        # near_rate = 0.08
        near_rate = 0.03
        self.edge_nearby_threshold = near_rate * rsphere

    def is_target_edge_nearby(self, fp_edge: TopoDS_Edge):
        current_dis = dist_point2shape(self, fp_edge)
        if current_dis < self.edge_nearby_threshold:
            return True
        else:
            return False

    def edge_nearby_cal(self):
        self.nearby_threshold_cal()

        edge_explorer = TopExp_Explorer(self.aligned_face, TopAbs_EDGE)

        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge = topods.Edge(edge)
            edge_explorer.Next()

            # # 过滤无效边
            # current_cur = BRep_Tool.Curve(edge)[0]
            # if not isinstance(current_cur, Geom_Curve):
            #     print('当前边无法处理，已跳过')
            #     continue

            if is_edge_useful(edge, self.edges_useful) and self.is_target_edge_nearby(edge):
                self.is_edge_nearby = 1
                return

    def get_save_str(self, is_contain_xyz=True):
        if is_contain_xyz:
            save_str = (f'{self.X()}\t{self.Y()}\t{self.Z()}\t' +
                        f'{self.mad.X()}\t{self.mad.Y()}\t{self.mad.Z()}\t' +
                        f'{self.is_edge_nearby}\t{self.pmt}\n')
        else:
            save_str = (f'{self.mad.X()}\t{self.mad.Y()}\t{self.mad.Z()}\t' +
                        f'{self.is_edge_nearby}\t{self.pmt}\n')

        return save_str


def face_type(face_occt: TopoDS_Face):
    """
    获取 occt 面的类型
    :param face_occt:
    :return: ['plane', 'cylinder', 'cone', 'sphere', 'freeform']
    """
    surface = BRep_Tool.Surface(face_occt)
    surface_type = surface.DynamicType()
    type_name = surface_type.Name()
    # 可能是 [
    # 'GeomPlate_Surface',
    # 'Geom_BSplineSurface',
    # 'Geom_BezierSurface',
    # 'Geom_RectangularTrimmedSurface',
    # 'Geom_ConicalSurface',
    # 'Geom_CylindricalSurface',
    # 'Geom_Plane',
    # 'Geom_SphericalSurface',
    # 'Geom_ToroidalSurface',
    # 'Geom_OffsetSurface'
    # 'Geom_SurfaceOfLinearExtrusion',
    # 'Geom_SurfaceOfRevolution',
    # 'ShapeExtend_CompositeSurface',
    # ]
    if type_name == 'Geom_Plane':
        return 'plane'
    elif type_name == 'Geom_CylindricalSurface':
        return 'cylinder'
    elif type_name == 'Geom_ConicalSurface':
        return 'cone'
    elif type_name == 'Geom_SphericalSurface':
        return 'sphere'
    else:
        return 'freeform'


def step_read_ctrl(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    if status == IFSelect_RetDone:
        step_reader.NbRootsForTransfer()
        step_reader.TransferRoot()
        model_shape = step_reader.OneShape()

        if model_shape.IsNull():
            raise ValueError('Empty STEP file')
        else:
            return step_reader.OneShape()

    else:
        raise ValueError('Cannot read the file')


def step_read_ocaf(filename):
    _shapes = []

    cafReader = STEPCAFControl_Reader()
    aDoc = TDocStd_Document("MDTV-XCAF")

    status = cafReader.ReadFile(filename)
    if status == IFSelect_RetDone:
        cafReader.Transfer(aDoc)
    else:
        raise ValueError('STET cannot be parsed:', filename)

    rootLabel = aDoc.Main()
    ShapeTool = XCAFDoc_DocumentTool.ShapeTool(rootLabel)

    aSeq = TDF_LabelSequence()
    ShapeTool.GetFreeShapes(aSeq)

    for i in range(aSeq.Length()):
        label = aSeq.Value(i + 1)
        loc = ShapeTool.GetLocation(label)
        part = TopoDS_Shape()
        ShapeTool.GetShape(label, part)

        if not loc.IsIdentity():
            part = part.Moved(loc)

        _shapes.append(part)

    return shapes_fuse(_shapes)


def step2stl(step_name, stl_name, deflection=0.1):
    shape_occ = step_read_ocaf(step_name)
    shapeocc2stl(shape_occ, stl_name, deflection)


def step2stl_batched_(dir_path, deflection=0.1):
    """
    将整个文件夹内的step转化为stl，stl保存在step同级文件夹
    """

    # 获取当前文件夹内全部step文件
    step_path_all = utils.get_allfiles(dir_path, 'step')
    n_step = len(step_path_all)

    for idx, c_step in enumerate(step_path_all):
        print(f'{idx} / {n_step}')
        stl_path = os.path.splitext(c_step)[0] + '.stl'

        step2stl(c_step, stl_path, deflection)


def shapes_fuse(shapes: list):
    if len(shapes) == 0:
        return TopoDS_Shape()  # 返回空形状

    elif len(shapes) == 1:
        return shapes[0]

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for shape in shapes:
        # if shape.ShapeType() == TopAbs_SOLID:
        builder.Add(compound, shape)

    return compound


def shape_area(shape_occ: TopoDS_Shape):
    """
    opencascade 中 TopoDS_Shape 的面积
    """
    props = GProp_PGProps()
    brepgprop.SurfaceProperties(shape_occ, props)
    return props.Mass()


def shapeocc2stl(shape_occ, save_path, deflection=0.1):
    save_path = os.path.abspath(save_path)

    # 清除三角面片
    # breptools.Clean(shape_occ)

    # 先进行三角面化
    mesh = BRepMesh_IncrementalMesh(shape_occ, deflection)
    mesh.Perform()
    assert mesh.IsDone()

    # 然后写stl
    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape_occ, save_path)


def is_edge_valid(fp_edge: TopoDS_Edge):
    """
    判断边是否有效，无效边不能参与计算
    """
    curve = BRep_Tool.Curve(fp_edge)[0]

    if isinstance(curve, Geom_Curve):
        return True

    else:
        return False


def is_edge_useful(edge: TopoDS_Edge, edges_useful: TopTools_IndexedMapOfShape):
    """
    判断边是有有用，即判断该边是否在有用边列表里
    """
    # assert edges_useful.Size() != 0

    if edges_useful.Contains(edge):
        return True
    else:
        return False


def is_point_in_shape(point: gp_Pnt, shape: TopoDS_Shape, tol: float = precision.Confusion()):
    dist2shape = dist_point2shape(point, shape)

    if dist2shape < tol:
        return True
    else:
        return False


def dist_point2shape(point: gp_Pnt, shape: TopoDS_Shape):
    vert = BRepBuilderAPI_MakeVertex(point)
    vert = vert.Shape()
    extrema = BRepExtrema_DistShapeShape(vert, shape)
    extrema.Perform()
    if not extrema.IsDone() or extrema.NbSolution() == 0:
        raise ValueError('计算点到形状距离失败')

    nearest_pnt = extrema.PointOnShape2(1)
    return point.Distance(nearest_pnt)


def edge_filter(edge_list: list):
    """
    查找所有非空边
    """
    valid_edges = []

    for edge in edge_list:
        if is_edge_valid(edge):
            valid_edges.append(edge)

    return valid_edges


def get_edges_useful(shape_occ: TopoDS_Shape):
    """
    从 occt 的 shape 中提取计算边缘有用的边，有用边包含：
    1. 在边界处不 G1 连续的面
    2. 有效边包含在边界处G1或更高连续性的面 2. 在边界处G1连续或更高连续性的面，但是基元类型不同或基元类型相同但参数不同
    """

    # 找到每条边对应的两个面
    def is_edge_in_face(fp_edge, fp_face):
        """
        判断某条边是否在某个面内
        """
        edge_exp = TopExp_Explorer(fp_face, TopAbs_EDGE)
        while edge_exp.More():
            edge_local = edge_exp.Current()
            edge_local = topods.Edge(edge_local)
            edge_exp.Next()

            if fp_edge.IsSame(edge_local):
                return True

        return False

    def find_adjfaces():
        """
        找到所有边对应的面
        这里只考虑每条边对应两个面的情况
        :return:
        """
        # key: edge, value: faces
        edge_adjface = {}

        for i in range(1, edges_all.Size() + 1):
            cdege = edges_all.FindKey(i)
            adjfaces = []

            # 遍历全部面
            face_exp = TopExp_Explorer(shape_occ, TopAbs_FACE)
            while face_exp.More():
                if len(adjfaces) == 2:
                    break

                face_local = face_exp.Current()
                face_local = topods.Face(face_local)
                face_exp.Next()

                if is_edge_in_face(cdege, face_local):
                    adjfaces.append(face_local)

            if len(adjfaces) == 2:
                edge_adjface[cdege] = adjfaces

        return edge_adjface

    def get_face_nornal_at_pnt(fp_point: gp_Pnt, fp_face: TopoDS_Face):
        """
        获取 fp_face 在 fp_point 处的法线
        """
        surf_local = BRep_Tool.Surface(fp_face)
        proj_local = GeomAPI_ProjectPointOnSurf(fp_point, surf_local)

        if proj_local.IsDone():
            fu, fv = proj_local.Parameters(1)
            face_props = GeomLProp_SLProps(surf_local, fu, fv, 1, precision.Confusion())
            normal_at = face_props.Normal()

            return normal_at

        else:
            raise ValueError('Can not perform projection')

    def is_edge_useful_by_adjface(fp_edge, adj_face1, adj_face2):
        """
        通过该边对应的两个面判断该边是否有效
        G1 或更高连续性情况下判定为无效
        :param fp_edge:
        :param adj_face1:
        :param adj_face2:
        :return:
        """
        # 取该边的中点
        acurve_info = BRep_Tool.Curve(fp_edge)
        acurve, p_start, p_end = acurve_info
        mid_pnt = acurve.Value((p_start + p_end) / 2.0)

        # 计算这个点到两个面的投影点的法线，法线共线则相切
        norm1 = get_face_nornal_at_pnt(mid_pnt, adj_face1)
        norm2 = get_face_nornal_at_pnt(mid_pnt, adj_face2)

        # 判断两向量是否共线：
        angle = norm1.Angle(norm2)
        prec_resolution = precision.Confusion() + 1e-5

        if angle < prec_resolution or abs(angle - np.pi) < prec_resolution:  # 法线共线，则边无用
            # print('发现无效边，角度差为：', min(angle, abs(angle - np.pi)))
            return False
        else:
            return True

    def is_edge_useful_by_commomobj(fp_edge, adj_face1, adj_face2):
        """
        通过判断边对应的两个面是否为同一实体，判断是否有效
        是同一实体判断为无效
        :param fp_edge:
        :param adj_face1:
        :param adj_face2:
        :return:
        """
        type1 = face_type(adj_face1)
        type2 = face_type(adj_face2)
        surf1 = BRep_Tool.Surface(adj_face1)
        surf2 = BRep_Tool.Surface(adj_face2)

        # 类型不同，直接判别为不同
        if type1 != type2:
            return True

        elif type1 == 'plane':
            pln1 = Geom_Plane.DownCast(surf1).Pln()
            pln2 = Geom_Plane.DownCast(surf2).Pln()

            # 提取法向量
            n1 = pln1.Axis().Direction()
            n2 = pln2.Axis().Direction()

            # 判断法向量是否平行（点乘绝对值接近 1）
            dot = abs(n1.Dot(n2))
            are_parallel = abs(dot - 1.0) < precision.Confusion()

            if are_parallel:
                # 任取 pln1 上的点，例如其位置点
                pnt = pln1.Location()
                # 计算该点到 pln2 的距离
                dist = pln2.Distance(pnt)

                # 两平面重合，边无效
                if dist < precision.Confusion():
                    return False
                else:
                    return True
            else:
                return True

        elif type1 == 'cylinder':
            c1 = Geom_CylindricalSurface.DownCast(surf1).Cylinder()
            c2 = Geom_CylindricalSurface.DownCast(surf2).Cylinder()

            # 半径不同，基元不同，边有效
            if abs(c1.Radius() - c2.Radius()) > precision.Confusion():
                return True

            # 方向
            dir1 = c1.Axis().Direction()
            dir2 = c2.Axis().Direction()
            if abs(abs(dir1.Dot(dir2)) - 1.0) > precision.Confusion():
                return True

            # 轴线重合
            p1 = c1.Axis().Location()
            dist = c2.Axis().Distance(p1)
            if dist > precision.Confusion():
                return True

            return False

        elif type1 == 'cone':

            c1 = Geom_ConicalSurface.DownCast(surf1).Cone()
            c2 = Geom_ConicalSurface.DownCast(surf2).Cone()
            # 半角
            if abs(c1.SemiAngle() - c2.SemiAngle()) > precision.Confusion():
                return True
            # 方向
            if abs(abs(c1.Axis().Direction().Dot(c2.Axis().Direction())) - 1.0) > precision.Confusion():
                return True
            # 顶点
            if c1.Apex().Distance(c2.Apex()) > precision.Confusion():
                return True

            return False

        elif type1 == 'sphere':
            s1 = Geom_SphericalSurface.DownCast(surf1).Sphere()
            s2 = Geom_SphericalSurface.DownCast(surf2).Sphere()
            if abs(s1.Radius() - s2.Radius()) > precision.Confusion():
                return True
            if s1.Location().Distance(s2.Location()) > precision.Confusion():
                return True

            return False

        # 自由曲面直接判定为面不同，边有效
        else:
            return True

    # 创建不重复容器
    edges_useful = TopTools_IndexedMapOfShape()
    edges_useful.Clear()
    edges_all = TopTools_IndexedMapOfShape()

    # 获取全部边
    edge_explorer = TopExp_Explorer(shape_occ, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        edge = topods.Edge(edge)
        edge_explorer.Next()

        try:
            if is_edge_valid(edge):
                edges_all.Add(edge)
        except:
            print('计算有效边时发现无法处理的边，已跳过')

    edge_face_pair = find_adjfaces()

    for cedge in edge_face_pair.keys():
        try:
            # if is_edge_useful_by_adjface(cedge, *edge_face_pair[cedge]):
            if is_edge_useful_by_commomobj(cedge, *edge_face_pair[cedge]):
                edges_useful.Add(cedge)
        except:
            print('计算有用边时发现无法处理的边，已跳过')

    return edges_useful


def get_point_aligned_face(model_occ: TopoDS_Shape, point: gp_Pnt, prec=0.1):
    """
    获取模型中该点所在面
    :param model_occ: 三维模型
    :param point: 目标点
    :param prec: 精度，可设为将 B-Rep 转化为 Mesh 时的误差
    :return: 目标点所在面
    """
    explorer = TopExp_Explorer(model_occ, TopAbs_FACE)

    while explorer.More():
        face = explorer.Current()
        face = topods.Face(face)
        explorer.Next()

        try:
            current_dist = dist_point2shape(point, face)
        except:
            print('无法计算点与当前面的距离，跳过当前面')
            continue

        if current_dist < prec + precision.Confusion():
            return face

    return None


def get_logger(name: str = 'log'):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log'), exist_ok=True)
    file_handler = logging.FileHandler(f'log/{name}-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def step2pcd(step_path, n_points, save_path, deflection=0.1, xyz_only=True):
    """
    将step模型转化为带约束的点云，需要先转化为 mesh
    :param step_path:
    :param n_points:
    :param save_path:
    :param deflection:
    :param xyz_only:
    :return:
    """

    # 生成 mesh
    tmp_stl = 'tmp/gen_pcd_cst.stl'
    step2stl(step_path, tmp_stl)
    vertex_matrix = mesh_proc.get_points_mslab(tmp_stl, n_points)

    if xyz_only:
        np.savetxt(save_path, vertex_matrix, fmt='%.6f', delimiter='\t')

    else:
        # 真实生成的点数，使用poisson_disk_sample得到的点数一般大于指定点数
        n_points_real = vertex_matrix.shape[0]

        model_occ = step_read_ocaf(step_path)
        model_area = shape_area(model_occ)
        edges_useful = get_edges_useful(model_occ)

        if edges_useful.Size() == 0:
            print('current model without Valid Edges')

        save_path = os.path.abspath(save_path)
        with open(save_path, 'w') as file_write:
            for i in tqdm(range(n_points_real), total=n_points_real):

                # 先找到该点所在面
                current_point = gp_Pnt(float(vertex_matrix[i, 0]), float(vertex_matrix[i, 1]),
                                       float(vertex_matrix[i, 2]))

                face_aligned = get_point_aligned_face(model_occ, current_point, deflection)

                if face_aligned is not None:
                    current_datapoint = Point3DForDataSet(current_point, face_aligned, model_area, edges_useful)
                    file_write.writelines(current_datapoint.get_save_str())

                else:
                    print(
                        f'find a point({current_point.X()}, {current_point.Y()}, {current_point.Z()}) without aligned face, skip')

    os.remove(tmp_stl)


def step2pcd_faceseg(step_path, n_points, save_path, deflection=0.1):
    """
    生成以面为分割的点云
    """
    # 生成 mesh
    tmp_stl = 'tmp/gen_pcd_cst.stl'
    step2stl(step_path, tmp_stl)
    vertex_matrix = mesh_proc.get_points_mslab(tmp_stl, n_points)

    # 真实生成的点数，使用poisson_disk_sample得到的点数一般大于指定点数
    n_points_real = vertex_matrix.shape[0]

    model_occ = step_read_ocaf(step_path)

    faceidx_dict = {}
    explorer = TopExp_Explorer(model_occ, TopAbs_FACE)

    face_count = 0
    while explorer.More():
        face = explorer.Current()
        face = topods.Face(face)
        explorer.Next()

        faceidx_dict[face] = face_count
        face_count += 1

    save_path = os.path.abspath(save_path)
    with open(save_path, 'w') as file_write:
        for i in tqdm(range(n_points_real), total=n_points_real):

            # 先找到该点所在面
            current_point = gp_Pnt(float(vertex_matrix[i, 0]), float(vertex_matrix[i, 1]), float(vertex_matrix[i, 2]))

            face_aligned = get_point_aligned_face(model_occ, current_point, deflection)

            if face_aligned is not None:
                aligned_faceidx = faceidx_dict[face_aligned]
                file_write.writelines(f'{float(vertex_matrix[i, 0])}\t{float(vertex_matrix[i, 1])}\t{float(vertex_matrix[i, 2])}\t{int(aligned_faceidx)}\n')

            else:
                print(
                    f'find a point({current_point.X()}, {current_point.Y()}, {current_point.Z()}) without aligned face, skip')

    os.remove(tmp_stl)


def step2pcd_batched(dir_path, n_points=2650, is_load_progress=True, xyz_only=False, deflection=0.1):
    """
    先整理成如下格式
    dir_path
    └─ raw
        ├─ car
        │   ├─ car0.stp
        │   ├─ car1.stp
        │   ├─ ...
        │   │
        │   ├─ small_car
        │   │   ├─ small_car0.stp
        │   │   ├─ small_car1.stp
        │   │   ├─ small_car2.stp
        │   │   ...
        │   │
        │   ├─ large_car
        │   │   ├─ large_car0.stp
        │   │   ├─ large_car1.stp
        │   │   ├─ large_car2.stp
        │   │   ...
        │   │
        │   ├─ car1.stp
        │   ...
        │
        ├─ plane
        │   ├─ plane0.stp
        │   ├─ plane1.stp
        │   ├─ plane2.stp
        │   ...
        │
        ...
    (car 文件夹下可存在子文件夹，但是，子文件夹内所有文件都将被归类为car)

    :param dir_path: 目标文件夹，分割使用
    :param n_points: 点云中点数
    :param is_load_progress: 是否从 json 文件中读取进度
    :param xyz_only: 是否仅输出 xyz 属性
    :param deflection: 三角面分割精度
    :return: void
    """

    # 获取日志记录句柄，日志文件名为 dir_path 名
    logger = get_logger(dir_path.split(os.sep)[-1])

    # 先在 dir_path 下生成文件夹 pointcloud
    pcd_dir = os.path.join(dir_path, 'pointcloud')
    os.makedirs(pcd_dir, exist_ok=True)

    # 然后获取raw文件夹下的全部类别，raw文件夹内的每个一级子文件夹名为一类
    path_allclasses = Path(os.path.join(dir_path, 'raw'))
    classes_all = utils.get_subdirs(path_allclasses)

    # 为每个类别在pointcloud文件夹下创建对应的文件夹
    utils.create_subdirs(pcd_dir, classes_all)

    # 先找到每个类别下的所有文件路径，便于后续程序中断可以继续
    class_file_all = {}
    for curr_class in classes_all:
        curr_read_save_paths = []

        trans_count = 0
        currclass_path = os.path.join(dir_path, 'raw', curr_class)
        for root, dirs, files in os.walk(currclass_path):
            for file in files:
                current_filepath = str(os.path.join(root, file))

                if utils.is_suffix_step(current_filepath):
                    file_name_pcd = str(trans_count) + '.txt'
                    trans_count += 1

                    current_savepath = os.path.join(dir_path, 'pointcloud', curr_class, file_name_pcd)
                    curr_read_save_paths.append((current_filepath, current_savepath))

        if len(curr_read_save_paths) != 0:
            class_file_all[curr_class] = curr_read_save_paths

    # 记录当前进度，程序中断后下次继续
    def get_progress():
        try:
            with open(filename_json, 'r') as file_json:
                progress = json.load(file_json)
        except:
            progress = {
                'dir_path': dir_path,
                'is_finished': False,
                'class_ind': 0,
                'instance_ind': 0
            }
            with open(filename_json, 'w') as file_json:
                json.dump(progress, file_json, indent=4)
        return progress

    def save_progress2json(progress_dict, class_ind, instance_ind):
        progress_dict['class_ind'] = class_ind
        progress_dict['instance_ind'] = instance_ind
        progress_dict['is_finished'] = False
        with open(filename_json, 'w') as file_json:
            json.dump(progress_dict, file_json, indent=4)

    def save_finish2json(progress_dict):
        progress_dict['is_finished'] = True
        with open(filename_json, 'w') as file_json:
            json.dump(progress_dict, file_json, indent=4)

    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
    os.makedirs(config_dir, exist_ok=True)
    filename_json = os.path.abspath(dir_path)
    filename_json = filename_json.replace(os.sep, '-').replace(':', '') + '.json'
    filename_json = os.path.join(config_dir, filename_json)

    trans_progress = get_progress()

    # 若从json文件中读取到当前文件夹已经转换完毕，则不转换该文件夹
    if trans_progress['is_finished']:
        print('从进度文件中读取到已完成该文件夹的转换，不再重复转换。若有需求请修改进度文件：' + filename_json)
        return

    # 从json文件中读取进度
    startind_class = 0
    startind_instance = 0
    if is_load_progress:
        if not trans_progress['is_finished'] and trans_progress['dir_path'] == dir_path:
            startind_class = trans_progress['class_ind']
            startind_instance = trans_progress['instance_ind']
            print('从文件读取进度：' + filename_json + f'- class_ind:{startind_class} - instance_ind:{startind_instance}')

    # 分别处理每个类别下的 STEP 文件
    trans_count_all = 0

    class_ind = startind_class
    for curr_class in itertools.islice(class_file_all.keys(), startind_class, None):

        if class_ind == startind_class:
            instance_ind = startind_instance
            startind_instance = startind_instance
        else:
            instance_ind = 0
            startind_instance = 0

        for curr_read_save_paths in itertools.islice(class_file_all[curr_class], startind_instance, None):
            try:
                save_progress2json(trans_progress, class_ind, instance_ind)

                print('当前转换：', curr_read_save_paths[0], f'类别-文件索引：{class_ind}-{instance_ind}', '转换进度：', trans_count_all)
                print('当前存储：', curr_read_save_paths[1], '时间：' + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                step2pcd(curr_read_save_paths[0], n_points, curr_read_save_paths[1], deflection, xyz_only)
            except:
                print('无法读取该STEP文件，已跳过：', curr_read_save_paths[0].encode('gbk', errors='ignore'))
                logger.info('跳过：' + curr_read_save_paths[0])
                continue

            instance_ind += 1
            trans_count_all += 1
        class_ind += 1

    save_finish2json(trans_progress)


def step2pcd_batched_(dir_path, n_points=2650, xyz_only=False, deflection=0.1):
    """
    将整个文件夹内的step转化为点云，点云保存在step同级文件夹
    """

    # 获取当前文件夹内全部step文件
    step_path_all = utils.get_allfiles(dir_path, 'step')
    n_step = len(step_path_all)

    for idx, c_step in enumerate(step_path_all):
        print(f'{idx} / {n_step}')
        pcd_path = os.path.splitext(c_step)[0] + '.txt'
        step2pcd(c_step, n_points, pcd_path, deflection, xyz_only)


def step2pcd_multi_batched(dirs_all: list):
    """
    多线程转换
    :param dirs_all: 目标文件夹列表
    :return:
    """
    threads_all = []

    for c_dir in dirs_all:
        c_thread = multiprocessing.Process(target=step2pcd_batched, args=(c_dir,))
        c_thread.start()
        threads_all.append(c_thread)

    for c_thread in threads_all:
        c_thread.join()


def assembly_filter(filename):
    """
    判断某个step模型是否为装配体
    :return: is assembly
    """
    cafReader = STEPCAFControl_Reader()
    aDoc = TDocStd_Document("MDTV-XCAF")

    status = cafReader.ReadFile(filename)
    if status == IFSelect_RetDone:
        cafReader.Transfer(aDoc)
    else:
        raise ValueError('STET cannot be parsed:', filename)

    rootLabel = aDoc.Main()
    ShapeTool = XCAFDoc_DocumentTool.ShapeTool(rootLabel)

    aSeq = TDF_LabelSequence()
    ShapeTool.GetFreeShapes(aSeq)

    shapes_count = 0
    for i in range(aSeq.Length()):
        label = aSeq.Value(i + 1)
        loc = ShapeTool.GetLocation(label)
        part = TopoDS_Shape()
        ShapeTool.GetShape(label, part)

        if not loc.IsIdentity():
            part = part.Moved(loc)

        if part.ShapeType() == TopAbs_SOLID:
            shapes_count += 1

    if shapes_count > 1:
        return True
    else:
        return False


def assemble_explode(filename):
    """
    将装配体的shape转化为一组零件shape
    :return: list
    """
    cafReader = STEPCAFControl_Reader()
    aDoc = TDocStd_Document("MDTV-XCAF")

    status = cafReader.ReadFile(filename)
    if status == IFSelect_RetDone:
        cafReader.Transfer(aDoc)
    else:
        raise ValueError('STET cannot be parsed:', filename)

    rootLabel = aDoc.Main()
    ShapeTool = XCAFDoc_DocumentTool.ShapeTool(rootLabel)

    aSeq = TDF_LabelSequence()
    ShapeTool.GetFreeShapes(aSeq)

    part_list = []

    for i in range(aSeq.Length()):
        label = aSeq.Value(i + 1)
        loc = ShapeTool.GetLocation(label)
        part = TopoDS_Shape()
        ShapeTool.GetShape(label, part)

        if not loc.IsIdentity():
            part = part.Moved(loc)

        if part.ShapeType() == TopAbs_SOLID:
            part_list.append(part)

    return part_list


def step_primitive_type_statistic(step_root):
    """
    统计一个step零件里的各类基元数
    :param step_root:
    :return:
    """
    shape_occ = step_read_ocaf(step_root)
    statis_res = {}

    face_explorer = TopExp_Explorer(shape_occ, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        face = topods.Face(face)
        face_explorer.Next()

        surface = BRep_Tool.Surface(face)
        surface_type = surface.DynamicType()
        type_name = surface_type.Name()

        if type_name in statis_res.keys():
            statis_res[type_name] += 1
        else:
            statis_res[type_name] = 1

    return statis_res


def step_primitive_type_statistic_batched(root):
    step_all = utils.get_allfiles(root, 'STEP')
    statis_all = Counter()

    for c_step in tqdm(step_all, total=len(step_all)):
        try:
            c_statis = step_primitive_type_statistic(c_step)
            statis_all = statis_all + Counter(c_statis)
        except:
            print(f'error occurred: {c_step}')

    print(statis_all)


def read_step_assembly(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != IFSelect_RetDone:
        raise RuntimeError("STEP 文件读取失败")

    reader.TransferRoots()
    shape = reader.Shape()

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(exp.Current())
        exp.Next()

    print(f"共读取 {len(solids)} 个零件")
    return solids


def test():
    stepfile = r''

    pass


if __name__ == '__main__':
    # step_assem = r'C:\Users\ChengXi\Desktop\螺纹连接.STEP'
    # solids = read_step_assembly(step_assem)

    # from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    # from OCC.Display.SimpleGui import init_display
    # from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    # # 创建两个立方体作为示例
    # shape1 = BRepPrimAPI_MakeBox(100, 100, 100).Shape()
    # shape2 = BRepPrimAPI_MakeBox(100, 100, 100).Shape()
    #
    # # 平移第二个立方体使它与第一个部分重叠
    # from OCC.Core.gp import gp_Trsf, gp_Vec
    # from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    #
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(50, 50, 50))
    # shape2_moved = BRepBuilderAPI_Transform(shape2, trsf, True).Shape()
    #
    # detect_interference_full([shape1, shape2])



    # step2pcd(r'C:\Users\ChengXi\Desktop\cylinder.STEP', 2500, r'C:\Users\ChengXi\Desktop\cylinder.txt', xyz_only=True)
    # step_primitive_type_statistic_batched(r'D:\document\DeepLearning\DataSet\STEPMillion\STEPMillion_0\raw')

    """
    <level1>
Start by creating a new coordinate system with Euler angles set to zero and a translation vector also set to zero. Next, draw a two-dimensional sketch on the first face. This sketch consists of a single loop made up of four lines. The first line starts at the origin (0.0, 0.0) and ends at (0.6, 0.0). The second line starts at (0.6, 0.0) and ends at (0.6, 0.375). The third line starts at (0.6, 0.375) and ends at (0.0, 0.375). Finally, the fourth line completes the loop by starting at (0.0, 0.375) and ending at the origin (0.0, 0.0). After drawing the sketch, apply a scale factor of 0.6 to the entire sketch. Ensure that the sketch remains aligned with the original coordinate system by rotating it using Euler angles set to zero and translating it using a vector set to zero. To transform the scaled two-dimensional sketch into a three-dimensional model, extrude the sketch 0.075 units along the normal direction. Do not extrude in the opposite direction of the normal. This operation will create a new solid body. The final dimensions of the rectangular block are a length of 0.5999999999999999 units, a width of 0.3749999999999999 units, and a height of 0.07499999999999998 units.
</level1>

<level2>
Create a rectangular block by starting with a two-dimensional sketch on a face. The sketch forms a rectangle with a length slightly less than 0.6 units and a width slightly less than 0.375 units. After drawing the sketch, scale it down to fit the desired dimensions. The scaled sketch is then transformed into a three-dimensional model by extruding it along the normal direction to a height slightly less than 0.075 units. The resulting solid body has a length, width, and height that closely match the specified dimensions, forming a compact rectangular block.
</level2>

<level3>
The design involves creating a small rectangular block. The block has a length and width that are roughly 0.6 and 0.375 units, respectively, and a height of about 0.075 units. The final shape is a simple, compact rectangular solid.
</level3>



###############################################################################
    <part_1>
### Construct a Rectangular Block
#### Create a New Coordinate System
- Set the Euler angles to 0.0, 0.0, 0.0.
- Set the translation vector to 0.0, 0.0, 0.0.

#### Draw a 2D Sketch
- **Face 1**
  - **Loop 1**
    - **Line 1**: Start at (0.0, 0.0) and end at (0.6, 0.0).
    - **Line 2**: Start at (0.6, 0.0) and end at (0.6, 0.375).
    - **Line 3**: Start at (0.6, 0.375) and end at (0.0, 0.375).
    - **Line 4**: Start at (0.0, 0.375) and end at (0.0, 0.0).

#### Scale the 2D Sketch
- Apply a scale factor of 0.6 to the sketch.

#### Transform the Scaled 2D Sketch into 3D
- Rotate the sketch using the Euler angles (0.0, 0.0, 0.0).
- Translate the sketch using the translation vector (0.0, 0.0, 0.0).

#### Extrude the 2D Sketch to Generate the 3D Model
- Extrude the sketch 0.075 units towards the normal.
- Do not extrude in the opposite direction of the normal.
- The operation creates a new solid body.

#### Final Dimensions
- Length: 0.5999999999999999
- Width: 0.3749999999999999
- Height: 0.07499999999999998
</part_1>
    
    """




    pass






