from onshape import on_utils
from onshape.OspGeomBase import OspPoint


class OspFace(object):
    """
    主要为了包含实体 id
    保存的实体包含全部信息，防止后续操作步骤对实体进行了更改，但是实体的 id 却不变
    """
    def __init__(self, bspline_face_info, edge_list: list, topo_id: str):
        self.id = topo_id
        self.edge_list = edge_list

        self.bspline_face = on_utils.make_bspline_face(bspline_face_info)

    def sample(self, num=50) -> list[OspPoint]:
        """
        为 face 的每个 edge 采样点
        """
        sample_points = []
        for edge in self.edge_list:
            edge_points = edge.sample(num)
            sample_points.append(edge_points)

        return sample_points

    def show(self):
        on_utils.show_occt_entity_list(self.bspline_face)


class OspBody(object):
    """
    主要为了包含实体 id
    保存的实体包含全部信息，防止后续操作步骤对实体进行了更改，但是实体的 id 却不变
    """
    def __init__(self, face_list: list[OspFace], topo_id: str):
        self.id = topo_id
        self.face_list = face_list

    def sample(self, num=50) -> list[OspPoint]:
        """
        显示每个面的每个边
        """
        sample_points = []
        for face in self.face_list:
            sample_points.extend(face.sample(num))

        return sample_points


    def show(self):
        """
        显示该实体
        """
        bspline_face_list = []

        for osp_face in self.face_list:
            bspline_face_list.append(osp_face.bspline_face)

        on_utils.show_occt_entity_list(bspline_face_list)

