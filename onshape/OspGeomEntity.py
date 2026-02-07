from onshape import on_utils


class OspFace(object):
    """
    主要为了包含实体 id
    保存的实体包含全部信息，防止后续操作步骤对实体进行了更改，但是实体的 id 却不变
    """
    def __init__(self, bspline_face_info, edge_list: list, topo_id: str):
        self.id = topo_id
        self.edge_list = edge_list

        self.bspline_face = on_utils.make_bspline_face(bspline_face_info)

    def show(self):
        on_utils.show_occt_entity_list(self.bspline_face)


class OspBody(object):
    """
    主要为了包含实体 id
    保存的实体包含全部信息，防止后续操作步骤对实体进行了更改，但是实体的 id 却不变
    """
    def __init__(self, bspline_face_list: list, topo_id: str):
        self.id = topo_id
        self.bspline_face_list = bspline_face_list

    def show(self):
        """
        显示该实体
        """
        on_utils.show_occt_entity_list(self.bspline_face_list)

