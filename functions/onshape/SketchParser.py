from functions.onshape.OspPoint import OspPoint
from functions.onshape.OspGeom import OspLine, OspCircle, SketchPlane


class SketchParser(object):
    """
    A parser for OnShape sketch feature list
    """
    def __init__(self, fea_item_ofs):
        self.feat_id = fea_item_ofs['message']['featureId'][0]
        self.feat_name = fea_item_ofs['message']['name']

        # 获取草图平面的 id
        self.sketch_plane_id = self.parse_sketch_plane_id(fea_item_ofs)

        # 先不获取，为了节省 request 数，最后一起获取
        self.sketch_plane = None

        # 获取草图的基元
        self.primitive_list = self.parse_sketch_primitives(fea_item_ofs)

    def load_sketch_plane(self, res_msg_val_ofs_item):
        self.sketch_plane = SketchPlane(res_msg_val_ofs_item)

        # 将草图中的元素都转换成 3d
        for i, c_prim in enumerate(self.primitive_list):
            self.primitive_list[i].map_to_3d(self.sketch_plane)

    @staticmethod
    def parse_sketch_plane_id(fea_item_ofs):
        """
        获取草图所在平面的id
        :param fea_item_ofs:
        :return:
        """
        param_list = fea_item_ofs['message']['parameters']

        sketch_plane_id = None
        for i, param_item in enumerate(param_list):
            param_msg = param_item['message']
            param_id = param_msg['parameterId']

            if param_id == 'sketchPlane' and 'queries' in param_msg:
                sketch_plane_id = param_msg['queries'][0]['message']['geometryIds'][0]
        return sketch_plane_id

    @staticmethod
    def parse_line_param(entity_item_msg_ofs):
        """
        从原始的信息中获取直线
        :param entity_item_msg_ofs:
        :return:
        """
        start_param = entity_item_msg_ofs['startParam']
        end_param = entity_item_msg_ofs['endParam']

        _entity_item_msg_geom_msg_ofs = entity_item_msg_ofs['geometry']['message']
        loc_x = _entity_item_msg_geom_msg_ofs['pntX']
        loc_y = _entity_item_msg_geom_msg_ofs['pntY']
        dir_x = _entity_item_msg_geom_msg_ofs['dirX']
        dir_y = _entity_item_msg_geom_msg_ofs['dirY']

        loc_line = OspPoint(loc_x, loc_y, 0, True)
        dir_line = OspPoint(dir_x, dir_y, 0, True)

        start = loc_line + start_param * dir_line
        end = loc_line + end_param * dir_line

        return OspLine(start, end, True)

    @staticmethod
    def parse_circle_param(entity_item_msg_ofs):
        """
        从原始的信息中获取圆
        :param entity_item_msg_ofs:
        :return:
        """
        entity_item_msg_geom_msg_ofs = entity_item_msg_ofs['geometry']['message']

        radius = entity_item_msg_geom_msg_ofs['radius']
        x_cen = entity_item_msg_geom_msg_ofs['xCenter']
        y_cen = entity_item_msg_geom_msg_ofs['yCenter']
        center = OspPoint(x_cen, y_cen, 0, True)

        return OspCircle(center, radius, None, True)

    @staticmethod
    def parse_sketch_primitives(fea_item_ofs):
        """
        解析草图中的几何元素
        :return:
        """

        entity_parsed_list = []

        entity_ofs_list = fea_item_ofs['message']['entities']
        for entity_item_ofs in entity_ofs_list:

            entity_item_msg_ofs = entity_item_ofs['message']
            entity_type = entity_item_msg_ofs['geometry']['typeName']
            entity_parsed = None

            if entity_type == 'BTCurveGeometryLine':
                entity_parsed = SketchParser.parse_line_param(entity_item_msg_ofs)

            elif entity_type == 'BTCurveGeometryCircle':
                entity_parsed = SketchParser.parse_circle_param(entity_item_msg_ofs)

            entity_parsed_list.append(entity_parsed)

        return entity_parsed_list






