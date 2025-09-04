# open cascade
import random

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
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

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
import shutil
from functools import partial
from multiprocessing import Pool
import uuid

# self
from processor import mesh_proc
from utils import utils


class Point3DForDataSet(gp_Pnt):
    """
    使用类处理点的额外属性
    """
    def __init__(self, pnt_loc: gp_Pnt, aligned_face: TopoDS_Face, prim_idx: int):
        super().__init__(pnt_loc.XYZ())

        aligned_surface = BRep_Tool.Surface(aligned_face)
        self.aligned_face = aligned_face
        self.prim_idx = prim_idx
        self.surf_adaptor = GeomAdaptor_Surface(aligned_surface)
        self.type_name = face_type(self.surf_adaptor)

        self.pmt = -1
        self.dir = gp_Vec(0.0, 0.0, -1.0)
        self.dim = -1.
        self.nor = gp_Vec(0.0, 0.0, -1.0)
        self.loc = gp_Pnt(0.0, 0.0, 0.0)

        self.pmt_dir_cal()
        self.dim_loc_cal()
        self.nor_cal()

    def pmt_dir_cal(self):
        """
        计算所在基元类型及主要方向
        :return:
        """
        if self.type_name == 'plane':
            self.pmt = 0
            self.dir = self.surf_adaptor.Plane().Axis().Direction().XYZ()

        elif self.type_name == 'cylinder':
            self.pmt = 1
            self.dir = self.surf_adaptor.Cylinder().Axis().Direction().XYZ()

        elif self.type_name == 'cone':
            self.pmt = 2
            self.dir = self.surf_adaptor.Cone().Axis().Direction().XYZ()

        elif self.type_name == 'sphere':
            self.pmt = 3

        elif self.type_name == 'freeform':
            self.pmt = 4

        else:
            raise TypeError('Undefined surf type')

        # 保证主方向唯一
        ax_x = self.dir.X()
        ax_y = self.dir.Y()
        ax_z = self.dir.Z()

        zero_lim = precision.Confusion()
        if ax_z < -zero_lim:  # z < 0 时, 反转
            self.dir *= -1.0
        elif abs(ax_z) <= zero_lim and ax_y < -zero_lim:  # z为零, y为负数, 反转
            self.dir *= -1.0
        elif abs(ax_z) <= zero_lim and abs(ax_y) <= zero_lim and ax_x < -zero_lim:  # z为零, y为零, x为负数, 反转
            self.dir *= -1.0
        else:
            # 无需反转
            pass

    def dim_loc_cal(self):
        """
        计算主尺寸和主位置
        :return:
        """
        if self.type_name == 'plane':  # 平面无主尺寸
            pln = self.surf_adaptor.Plane()
            a, b, c, d = pln.Coefficients()

            ad = a * d
            bd = b * d
            cd = c * d
            length = (a * a + b * b + c * c) ** 0.5

            perpendicular_foot = gp_Pnt(- ad / length, - bd / length, - cd / length)
            self.loc = perpendicular_foot

        elif self.type_name == 'cylinder':
            cy_surf = self.surf_adaptor.Cylinder()
            self.dim = cy_surf.Radius()

            axis = cy_surf.Axis()
            cyloc = axis.Location()  # 轴线上一点 (gp_Pnt)
            cydir = axis.Direction()  # 轴线方向 (gp_Dir)

            # 投影长度 (点积)
            t = - gp_Vec(cyloc.XYZ()).Dot(gp_Vec(cydir))

            # 垂足坐标
            perpendicular_foot = gp_Pnt(cyloc.XYZ() + cydir.XYZ().Multiplied(t))
            self.loc = perpendicular_foot

        elif self.type_name == 'cone':
            cone_surf = self.surf_adaptor.Cone()
            self.dim = cone_surf.SemiAngle()
            self.loc = cone_surf.Apex()

        elif self.type_name == 'sphere':
            sph_surf = self.surf_adaptor.Sphere()
            self.dim = sph_surf.Radius()
            self.loc = sph_surf.Location()

        elif self.type_name == 'freeform':  # 自由面无主尺寸和主位置
            pass

        else:
            raise TypeError('error surface type')

    def nor_cal(self):
        """
        计算该点处的法线
        :return:
        """
        self.nor = normal_at(self, self.aligned_face)

    def get_save_str(self, is_contain_xyz=True):
        if is_contain_xyz:
            save_str = (f'{self.X()} {self.Y()} {self.Z()} ' +  # 坐标
                        f'{self.pmt} ' +  # 基元类型
                        f'{self.dir.X()} {self.dir.Y()} {self.dir.Z()} ' +  # 主方向
                        f'{self.dim} ' +  # 主尺寸
                        f'{self.nor.X()} {self.nor.Y()} {self.nor.Z()} ' +  # 法线
                        f'{self.loc.X()} {self.loc.Y()} {self.loc.Z()} ' +  # 主位置
                        f'{self.prim_idx}\n')  # 基元索引
        else:
            save_str = (f'{self.pmt} ' +  # 基元类型
                        f'{self.dir.X()} {self.dir.Y()} {self.dir.Z()} ' +  # 主方向
                        f'{self.dim} ' +  # 主尺寸
                        f'{self.nor.X()} {self.nor.Y()} {self.nor.Z()} ' +  # 法线
                        f'{self.loc.X()} {self.loc.Y()} {self.loc.Z()} ' +  # 主位置
                        f'{self.prim_idx}\n')  # 基元索引

        return save_str

    def get_save_data(self, is_contain_xyz=True):
        if is_contain_xyz:
            save_data = (self.X(), self.Y(), self.Z(),  # 坐标
                         self.pmt,  # 基元类型
                         self.dir.X(), self.dir.Y(), self.dir.Z(),  # 主方向
                         self.dim,  # 主尺寸
                         self.nor.X(), self.nor.Y(), self.nor.Z(),  # 法线
                         self.loc.X(), self.loc.Y(), self.loc.Z(),  # 主位置
                         self.prim_idx)  # 基元索引
        else:
            save_data = (self.pmt,  # 基元类型
                         self.dir.X(), self.dir.Y(), self.dir.Z(),  # 主方向
                         self.dim,  # 主尺寸
                         self.nor.X(), self.nor.Y(), self.nor.Z(),  # 法线
                         self.loc.X(), self.loc.Y(), self.loc.Z(),  # 主位置
                         self.prim_idx)  # 基元索引

        return save_data


def normalize_shape_to_unit_cube(shape):
    """
    将 OCC 的 TopoDS_Shape 质心移动到原点，三轴范围移动到 [-1, 1]
    :param shape:
    :return:
    """

    # Step 1: 获取包围盒
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()

    # Step 2: 平移到原点
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    cz = (zmin + zmax) * 0.5
    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(gp_Vec(-cx, -cy, -cz))
    shape_centered = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()

    # Step 3: 缩放到 [-1,1]^3
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    scale = 1.0 / max(dx, dy, dz)
    trsf_scale = gp_Trsf()
    trsf_scale.SetScale(gp_Pnt(0, 0, 0), scale)
    shape_normalized = BRepBuilderAPI_Transform(shape_centered, trsf_scale, True).Shape()

    return shape_normalized


def get_shape_faces(model_occ: TopoDS_Shape):
    """
    获取 OCCT 实体中的全部面
    :param model_occ:
    :return:
    """
    explorer = TopExp_Explorer(model_occ, TopAbs_FACE)

    face_all = []
    while explorer.More():
        face = explorer.Current()
        face = topods.Face(face)
        face_all.append(face)
        explorer.Next()

    return face_all


def face_type(adaptor: GeomAdaptor_Surface):
    """
    获取 occt 面的类型
    :param adaptor:
    :return: ['plane', 'cylinder', 'cone', 'sphere', 'freeform']
    """

    try:
        adaptor.Plane()
        return 'plane'
    except:
        pass

    try:
        adaptor.Cylinder()
        return 'cylinder'
    except:
        pass

    try:
        adaptor.Cone()
        return 'cone'
    except:
        pass

    try:
        adaptor.Sphere()
        return 'sphere'
    except:
        pass

    return 'freeform'

    # else if (!adaptor.Cylinder().IsNull()) {
    # std::
    #     cout << "This face is a Cylinder" << std::endl;
    # }
    # else if (!adaptor.Cone().IsNull()) {
    # std::
    #     cout << "This face is a Cone" << std::endl;
    # }
    # else if (!adaptor.Sphere().IsNull()) {
    # std::
    #     cout << "This face is a Sphere" << std::endl;
    # }
    # else if (!adaptor.Torus().IsNull()) {
    # std::
    #     cout << "This face is a Torus" << std::endl;
    # }
    # else if (!adaptor.Bezier().IsNull()) {
    # std::
    #     cout << "This face is a Bezier Surface" << std::endl;
    # }
    # else if (!adaptor.BSpline().IsNull()) {
    # std::
    #     cout << "This face is a B-Spline Surface" << std::endl;
    # }
    # else {
    #     std:: cout << "Other surface type (Offset, Extrusion, Revolution, etc.)" << std::endl;
    # }


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


def face_type_name(tds_face: TopoDS_Face):
    surface = BRep_Tool.Surface(tds_face)
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

    return type_name


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


def step2stl(step_name, stl_name, deflection=0.1, is_normalize=False):
    """
    将step文件转化为stl文件
    :param step_name:
    :param stl_name:
    :param deflection:
    :param is_normalize: 是否将step模型标准化 (中心平移到原点，范围缩放到 [-1, 1]^3)
    :return:
    """
    shape_occ = step_read_ocaf(step_name)

    if is_normalize:
        shape_occ = normalize_shape_to_unit_cube(shape_occ)

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


def shapeocc2step(shape, filename):
    writer = STEPControl_Writer()

    # 添加要写出的shape
    writer.Transfer(shape, STEPControl_AsIs)

    # 写文件
    status = writer.Write(filename)
    if status == IFSelect_RetDone:
        print(f"STEP file saved: {filename}")
    else:
        print("Error: failed to write STEP file")


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


def normal_at(point: gp_Pnt, face: TopoDS_Face):
    """
    获取 face 在 point 处的法线
    """
    surf_local = BRep_Tool.Surface(face)
    proj_local = GeomAPI_ProjectPointOnSurf(point, surf_local)
    is_reversed = (face.Orientation() == TopAbs_REVERSED)

    if proj_local.IsDone():
        fu, fv = proj_local.Parameters(1)
        face_props = GeomLProp_SLProps(surf_local, fu, fv, 1, precision.Confusion())
        normal = face_props.Normal()

        if is_reversed:
            normal.Reverse()

        return normal

    else:
        raise ValueError('Can not perform projection')


def get_point_aligned_face(model_occ: TopoDS_Shape, point: gp_Pnt, prec=0.1):
    """
    获取模型中该点所在面
    :param model_occ: 三维模型
    :param point: 目标点
    :param prec: 精度，可设为将 B-Rep 转化为 Mesh 时的误差
    :return: 目标点所在面
    """
    explorer = TopExp_Explorer(model_occ, TopAbs_FACE)

    c_idx = -1
    while explorer.More():
        face = explorer.Current()
        face = topods.Face(face)
        explorer.Next()
        c_idx += 1

        try:
            current_dist = dist_point2shape(point, face)
        except:
            print('无法计算点与当前面的距离，跳过当前面')
            continue

        if current_dist <= prec + precision.Confusion():
            return face, c_idx

    return None, None


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


def step2pcd(step_path, save_path, n_points, deflection=0.1, xyz_only=False, using_tqdm=True, print_log=True, is_normalize=True, over_rate=0.025):
    """
    将step模型转化为带约束的点云，需要先转化为 mesh
    :param step_path:
    :param save_path:
    :param n_points:
    :param deflection:
    :param xyz_only:
    :param using_tqdm: 是否使用进度条
    :param print_log:
    :param is_normalize:
    :param over_rate: 初始采样点数超过指定点数的比例
    :return:
    """

    # 生成 mesh
    n_itera = 0
    n_csample = n_points

    shape_occ = step_read_ocaf(step_path)
    if is_normalize:
        shape_occ = normalize_shape_to_unit_cube(shape_occ)

    tmp_stl = f'tmp/{uuid.uuid4()}.stl'  # 生成不重复的临时文件
    shapeocc2stl(shape_occ, tmp_stl, deflection)

    while True:
        vertex_matrix = mesh_proc.get_points_mslab(tmp_stl, n_csample)
        n_real_sampled = vertex_matrix.shape[0]

        if n_itera >= 100:
            raise ValueError('arrive max iteration, point number can not satisfy')
        elif n_real_sampled >= n_points * (1.0 + over_rate):
            break
        else:
            n_itera += 1
            n_csample = int(n_csample * 1.05)

    os.remove(tmp_stl)

    n_real_saved = 0
    if xyz_only:
        np.savetxt(save_path, vertex_matrix, fmt='%.6f')
        n_real_saved = vertex_matrix.shape[0]

    else:
        # 真实生成的点数，使用poisson_disk_sample得到的点数一般大于指定点数
        n_points_real = vertex_matrix.shape[0]

        save_path = os.path.abspath(save_path)
        with open(save_path, 'w') as file_write:
            for i in tqdm(range(n_points_real), total=n_points_real, disable=not using_tqdm):

                try:
                    # 先找到该点所在面
                    current_point = gp_Pnt(float(vertex_matrix[i, 0]),
                                           float(vertex_matrix[i, 1]),
                                           float(vertex_matrix[i, 2]))

                    face_aligned, idx = get_point_aligned_face(shape_occ, current_point, deflection)

                    if face_aligned is not None:
                        current_datapoint = Point3DForDataSet(current_point, face_aligned, idx)
                        file_write.writelines(current_datapoint.get_save_str())
                        n_real_saved += 1

                    elif print_log:
                        print(f'find a point({current_point.X()}, {current_point.Y()}, {current_point.Z()}) without aligned face, skip')

                except:
                    if print_log:
                        print(
                            f'find a point({current_point.X()}, {current_point.Y()}, {current_point.Z()}) without aligned face, skip')

    return n_real_saved


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


def step2pcd_abc(dir_path, n_points=2650, is_load_progress=True, xyz_only=False, deflection=0.1):
    """
    用于 ABC 数据集里的文件转化

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
        ├─ airplane
        │   ├─ airplane0.stp
        │   ├─ airplane1.stp
        │   ├─ airplane2.stp
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
                step2pcd(curr_read_save_paths[0], curr_read_save_paths[1], n_points, deflection, xyz_only)
            except:
                print('无法读取该STEP文件，已跳过：', curr_read_save_paths[0].encode('gbk', errors='ignore'))
                logger.info('跳过：' + curr_read_save_paths[0])
                continue

            instance_ind += 1
            trans_count_all += 1
        class_ind += 1

    save_finish2json(trans_progress)


def step2pcd_batched_multi_processing_wrapper(c_step, source_dir, target_dir, n_points, deflection, xyz_only):
    pcd_path = c_step.replace(source_dir, target_dir)
    pcd_path = os.path.splitext(pcd_path)[0] + '.txt'

    try:
        step2pcd(c_step, pcd_path, n_points, deflection, xyz_only, False, False, True)
    except:
        print(f'cannot convert this STEP file: {c_step}.')


def step2pcd_batched(source_dir, target_dir, n_points=2000, deflection=0.1, xyz_only=False, workers=4):
    """
    将 source_dir 下的 STEP 转化为 target_dir 下的点云，两者具备相同的目录层级结构
    :param source_dir:
    :param target_dir:
    :param n_points:
    :param deflection:
    :param xyz_only:
    :param workers: 进程数
    :return:
    """
    # 先在 target_path 下创建相同的目录结构
    os.makedirs(target_dir, exist_ok=True)

    # 清空target_dir
    print('clear dir: ', target_dir)
    shutil.rmtree(target_dir)

    utils.create_tree_like(source_dir, target_dir)
    files_all = utils.get_allfiles(source_dir, 'step')

    # 获取全部点云保存路径


    work_func = partial(
        step2pcd_batched_multi_processing_wrapper,
        source_dir=source_dir,
        target_dir=target_dir,
        n_points=n_points,
        deflection=deflection,
        xyz_only=xyz_only
    )

    with Pool(processes=workers) as pool:
        _ = list(tqdm(
            pool.imap(work_func, files_all),
            total=len(files_all),
            desc='STEP to point cloud')
        )


def step2pcd_batched_(dir_path, n_points=2000, deflection=0.1, xyz_only=False):
    """
    就地将整个文件夹内的step转化为点云，点云保存在step同级文件夹
    """

    # 获取当前文件夹内全部step文件
    step_path_all = utils.get_allfiles(dir_path, 'step')
    n_step = len(step_path_all)

    for idx, c_step in enumerate(step_path_all):
        print(f'{idx} / {n_step}')
        pcd_path = os.path.splitext(c_step)[0] + '.txt'

        try:
            step2pcd(c_step, pcd_path, n_points, deflection, xyz_only)
        except:
            print(f'cannot convert this STEP file: {c_step}.')


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
    step_dir = utils.get_allfiles(r'D:\document\DeepLearning\paper_draw\AttrVis_ABC', 'step')
    valid_name = ['Geom_BSplineSurface', 'Geom_BezierSurface', 'Geom_ConicalSurface', 'Geom_SphericalSurface', 'Geom_Plane', 'Geom_CylindricalSurface']

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

    # for c_step in step_dir:

    display, start_display, add_menu, add_function_to_menu = init_display()

    c_shape = step_read_ocaf(r'D:\document\DeepLearning\paper_draw\AttrVis_ABC\00028000\00028000_29d91d68447846af91317508_step_002.step')

    all_faces = get_shape_faces(c_shape)

    for cc_shape in all_faces:
        cc_name = face_type_name(cc_shape)
        if cc_name not in valid_name:

            aligned_surface = BRep_Tool.Surface(cc_shape)
            surf_adaptor = GeomAdaptor_Surface(aligned_surface)

            cc_name_2 = face_type(surf_adaptor)

            display.DisplayShape(cc_shape, update=True)

            display.FitAll()

            # 开启交互窗口
            start_display()



            print(cc_name + '-' + cc_name_2 + '-')

            exit(0)

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






