from collections import OrderedDict


def parse_feature_param(feat_item_msg_param_ofs):
    param_dict = {}
    for i, param_item in enumerate(feat_item_msg_param_ofs):
        param_msg = param_item['message']
        param_id = param_item['message']['parameterId']

        if 'queries' in param_msg:
            param_value = []
            for j in range(len(param_msg['queries'])):
                param_value.extend(param_msg['queries'][j]['message']['geometryIds'])

        elif 'expression' in param_msg:
            param_value = param_msg['expression']

        elif 'value' in param_msg:
            param_value = param_msg['value']

        elif 'items' in param_msg:
            param_value = []
            j_items_ofs = param_msg['items']

            for j_item in j_items_ofs:
                k_param_ofs = j_item['message']['parameters']
                for k_item in k_param_ofs:
                    l_query_ofs = k_item['message']['queries']
                    for l_query in l_query_ofs:
                        param_value.extend(l_query['message']['geometryIds'])

        else:
            raise NotImplementedError('param_msg:\n{}'.format(param_msg))

        param_dict[param_id] = param_value
    return param_dict


class Extrude(object):
    """
    拉伸命令
    目前仅支持: [单侧拉伸, 两侧对称拉伸], 不支持两侧分别进行距离不等的拉伸,拉伸到实体\曲面\顶点
    """
    def __init__(self, region, depth, direction, draft_angle, is_symmetric, is_opposite_dir):
        self.region = region
        self.depth = depth
        self.direction = direction
        self.draft_angle = draft_angle
        self.is_symmetric = is_symmetric
        self.is_opposite_dir = is_opposite_dir

        # 所需的全部实体 id
        self.required_geo = self.region

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取拉伸参数

        可获得的参数如下:

        key = feat_item_msg_param_item["message"]["parameterId"]

        bodyType: SOLID, .生成的实体类型
        operationType: "NEW", "REMOVE", "ADD", "INTERSECT". 操作类型，新增实体、或者删除实体
        surfaceOperationType: "NEW", .曲面操作类型，可先不考虑
        entities: ["queries"][0]["message"]["geometryIds"] = ["JGK"]. 拉伸涉及的区域 ID
        midplane: false. 未知
        thickness1: "0.25 in". 未知
        thickness2: "0 in". 未知
        thickness: "0.25 in". 未知
        endBound: "BLIND", "SYMMETRIC", . BLIND 为单侧指定距离拉伸，SYMMETRIC 为两侧对称拉伸
        oppositeDirection: false. 是否反向拉伸
        depth: "0.5 in" 总的拉伸深度
        endBoundEntityFace: 拉伸到曲面的曲面实体 ID？例如 JDC？
        endBoundEntityBody: 拉伸到实体的实体？
        endBoundEntityVertex: 拉伸到角点的角点实体？
        hasOffset: false. 拉伸时是否沿着草图基准面法线方向偏移一段距离？
        offsetDistance: "1 in". 偏移距离
        offsetOppositeDirection: 偏移是否反向
        hasExtrudeDirection: false. 是否指定拉伸方向
        extrudeDirection: . 具体的拉伸方向
        startOffset: false. 未知
        startOffsetBound: "BLIND". 未知
        startOffsetDistance: "1 in". 未知
        startOffsetOppositeDirection: false. 未知
        startOffsetEntity: . 未知
        symmetric: false. 是否对称拉伸
        hasDraft: false. 是否有拔模斜度
        draftAngle: "expression": "3 deg", "isInteger": false. 拔模斜度具体角度
        draftPullDirection: false.未知
        hasSecondDirection: false. 是否有第二个拉伸方向
        secondDirectionBound: "BLIND". 第二拉伸方向的结束面，BLIND: 拉伸到指定深度
        secondDirectionOppositeDirection: true
        secondDirectionDepth: "1 in". 第二拉伸方向的拉伸深度
        secondDirectionBoundEntityFace: [].
        secondDirectionBoundEntityBody: [].
        secondDirectionBoundEntityVertex: [].
        hasSecondDirectionOffset: false.
        secondDirectionOffsetDistance: "1 in".
        secondDirectionOffsetOppositeDirection: false. 第二拉伸方向是否反向
        hasSecondDirectionDraft: false. 第二拉伸方向是否有拔模斜度
        secondDirectionDraftAngle: "expression": "3 deg", "isInteger": false.
        secondDirectionDraftPullDirection: false. 未知
        defaultScope: false. 未知
        booleanScope: ["queries"][0]["message"]["geometryIds"] = ["JHD"]. 未知
        defaultSurfaceScope: true. 未知
        booleanSurfaceScope: . 未知

        :param feat_item_ofs:
        :return:
        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])
        # if 'hasOffset' in param_dict and param_dict['hasOffset'] is True:
        #     raise NotImplementedError("extrude with offset not supported: {}".format(param_dict['hasOffset']))

        region = param_dict['entities']  # 引用的几何体，例如 JDC，具体定义后续统一获取

        # 拉伸深度
        depth = param_dict['depth']  # TODO：字符串需要处理

        # 拉伸方向
        if param_dict['hasExtrudeDirection']:
            direction = None
        else:
            direction = False

        # 拔模斜度
        if param_dict['hasDraft']:
            draft_angle = param_dict['draftAngle']
        else:
            draft_angle = False

        # 是否两侧对称拉伸
        is_symmetric = param_dict['symmetric']

        # 拉伸方向是否与面法向反向
        is_opposite_dir = param_dict['oppositeDirection']

        return cls(region, depth, direction, draft_angle, is_symmetric, is_opposite_dir)


class Revolve(object):
    """
    仅支持指定角度的旋转, 不支持旋转到指定实体
    """
    def __init__(self, region, axis, angle, is_symmetric, is_opposite_dir):
        self.region = region
        self.axis = axis
        self.angle = angle
        self.is_symmetric = is_symmetric
        self.is_opposite_dir = is_opposite_dir

        # 所需的全部实体 id
        self.required_geo = self.region + self.axis

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取旋转参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        bodyType: "SOLID"
        operationType: "ADD"
        surfaceOperationType: "NEW". 未知
        entities: ["queries"][0]["message"]["geometryIds"] = ["JJC"]. 旋转涉及的区域 ID
        surfaceEntities: . 未知
        wallShape: . 未知
        midplane: false. 未知
        thickness1: "0.25 in". 未知
        flipWall: false. 未知
        thickness2: "0 in". 未知
        thickness: "0.25 in". 未知
        axis: ["queries"][0]["message"]["geometryIds"] = ["JHF"]. 旋转轴涉及的区域 ID
        fullRevolve: false. 是否整周旋转
        endBound: "BLIND". 结束面类型, "BLIND" 为指定角度的旋转
        oppositeDirection: false. 是否沿法线反向旋转
        angle: "120 deg". 旋转角度
        symmetric: false. 是否对称旋转
        endBoundEntityFace: .
        endBoundEntityBody: .
        endBoundEntityVertex: .
        endBoundHasOffset: false. 未知
        endBoundOffset: "0 deg".
        endBoundOffsetFlip: false.
        hasStartBound: false.
        startBound: "BLIND".
        angleBack: "expression": "330 deg", "isInteger": false.
        startOppositeDirection: false.
        startBoundEntityFace: .
        startBoundEntityBody: .
        startBoundEntityVertex: .
        startBoundHasOffset: false.
        startBoundOffset: "0 deg".
        startBoundOffsetFlip: false.
        defaultScope: false.
        booleanScope: ["queries"][0]["message"]["geometryIds"] = ["JHD"].
        defaultSurfaceScope: true.
        booleanSurfaceScope: .

        :param feat_item_ofs:
        :return:
        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])

        region = param_dict['entities']  # 引用的几何体，例如 JDC，具体定义后续统一获取

        axis = param_dict['axis']

        if param_dict['fullRevolve']:
            angle = '360 deg'
        else:
            angle = param_dict['angle']

        is_symmetric = param_dict['symmetric']

        is_opposite_dir = param_dict['oppositeDirection']

        return cls(region, axis, angle, is_symmetric, is_opposite_dir)


class Sweep(object):
    """
    仅支持指定路径的扫描
    """
    def __init__(self, region, path):
        self.region = region
        self.path = path

        # 所需的全部实体 id
        self.required_geo = self.region + self.path

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取扫描参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        bodyType: "SOLID".
        operationType: "ADD".
        surfaceOperationType: "NEW".
        - profiles: ["queries"][0]["message"]["geometryIds"] = ["JPC", "JPG"].
        surfaceProfiles: .
        wallShape: .
        midplane: .
        thickness1: "0.25 in".
        flipWall: false.
        thickness2: "0 in".
        thickness: "0.25 in".
        - path: ["queries"][0]["message"]["geometryIds"] = ["JHt"], ["queries"][1]["message"]["geometryIds"] = ["JHd"]
        profileControl: "NONE".
        lockFaces: .
        lockDirectionQuery: .
        trimEnds: true.
        defaultScope: false.
        booleanScope: ["queries"][0]["message"]["geometryIds"] = ["JHD"]
        defaultSurfaceScope: true
        booleanSurfaceScope: .

        :param feat_item_ofs:
        :return:
        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])

        region = param_dict['profiles']
        path = param_dict['path']

        return cls(region, path)


class Loft(object):
    """
    只支持几个轮廓，不支持带引导线的放样
    """

    def __init__(self, sections):
        self.sections = sections

        # 所需的全部实体 id
        self.required_geo = self.sections

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        bodyType: "SOLID".
        operationType: "ADD".
        surfaceOperationType: "NEW".
        - sheetProfilesArray: ["items"][0]["message"]["parameters"][0]["message"]["parameterId"] = "sheetProfileEntities".
                              ["items"][0]["message"]["parameters"][0]["message"]["queries"][0]["message"]["geometryIds"] = ["JTC"].
                              ["items"][1]["message"]["parameters"][0]["message"]["parameterId"] = "sheetProfileEntities".
                              ["items"][1]["message"]["parameters"][0]["message"]["queries"][0]["message"]["geometryIds"] = ["JKK"]

        wireProfilesArray: .
        midplane: false.
        thickness1: false.
        flipWall: false.
        thickness2: "expression": "0 in", "isInteger": false.
        thickness: "expression": "0.25 in", "isInteger": false.
        startCondition: "NORMAL_TO_PROFILE". 起始面状态：垂直于轮廓
        adjacentFacesStart: .
        startMagnitude: "expression": "2", "isInteger": false
        startDirection: .
        endCondition: "TANGENT_TO_PROFILE". 终止面状态：与轮廓相切
        adjacentFacesEnd: .
        endMagnitude: "expression": "1.2", "isInteger": false.
        endDirection: .
        trimProfiles: false.
        addGuides: false.
        guidesArray: .
        trimGuidesByProfiles: true.
        addSections: false.
        spine: .
        sectionCount: "expression": "5", "isInteger": true
        matchConnections: false.
        connections: .
        makePeriodic: false.
        showIsocurves: false.
        curveCount: "expression": "10", "isInteger": true.
        trimEnds: true.
        defaultScope: false.
        booleanScope: ["queries"][0]["message"]["geometryIds"] = ["JHD"].
        defaultSurfaceScope: true.
        booleanSurfaceScope: .

        :param feat_item_ofs:
        :return:
        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])
        sections = param_dict['sheetProfilesArray']
        return cls(sections)


EXTENT_TYPE_MAP = {'BLIND': 'OneSideFeatureExtentType',
                   'SYMMETRIC': 'SymmetricFeatureExtentType'}

OPERATION_MAP = {'NEW': 'NewBodyFeatureOperation',
                 'ADD': 'JoinFeatureOperation',
                 'REMOVE': 'CutFeatureOperation',
                 'INTERSECT': 'IntersectFeatureOperation'}


def xyz_list2dict(xyz):
    return OrderedDict({'x': xyz[0], 'y': xyz[1], 'z': xyz[2]})


class FeatureListParser(object):
    def __init__(self, client, did, wid, eid, feature_list=None):
        """
        A parser for OnShape feature list (construction sequence)
        :param client:
        :param did:
        :param wid:
        :param eid:
        :param feature_list:
        """
        self.client = client

        self.did = did
        self.wid = wid
        self.eid = eid

        if feature_list is None:
            print('get feature_list by onshape_api request')
            self.feature_list = self.client.get_features(did, wid, eid).json()
        else:
            print('load feature_list')
            self.feature_list = feature_list

        self.profile2sketch = {}

    @staticmethod
    def parse_feature_param(feat_param_data):
        param_dict = {}
        for i, param_item in enumerate(feat_param_data):
            param_msg = param_item['message']
            param_id = param_msg['parameterId']

            if 'queries' in param_msg:
                param_value = []
                for j in range(len(param_msg['queries'])):
                    param_value.extend(param_msg['queries'][j]['message']['geometryIds'])

            elif 'expression' in param_msg:
                param_value = param_msg['expression']

            elif 'value' in param_msg:
                param_value = param_msg['value']

            else:
                raise NotImplementedError('param_msg:\n{}'.format(param_msg))

            param_dict[param_id] = param_value
        return param_dict

    def _parse_sketch(self, feature_data):
        sket_parser = SketchParser(self.client, feature_data, self.did, self.wid, self.eid)
        save_dict = sket_parser.parse_to_fusion360_format()
        return save_dict

    def _expr2meter(self, expr):
        return self.client.expr2meter(self.did, self.wid, self.eid, expr)

    def _locate_sketch_profile(self, geo_ids):
        return [{"profile": k, "sketch": self.profile2sketch[k]} for k in geo_ids]

    def _parse_extrude(self, feature_data):
        """
        解析拉伸参数
        :param feature_data:
        :return:
        """
        param_dict = self.parse_feature_param(feature_data['parameters'])
        if 'hasOffset' in param_dict and param_dict['hasOffset'] is True:
            raise NotImplementedError("extrude with offset not supported: {}".format(param_dict['hasOffset']))

        entities = param_dict['entities']  # geometryIds for target face
        profiles = self._locate_sketch_profile(entities)

        extent_one = self._expr2meter(param_dict['depth'])
        if param_dict['endBound'] == 'SYMMETRIC':
            extent_one = extent_one / 2

        if 'oppositeDirection' in param_dict and param_dict['oppositeDirection'] is True:
            extent_one = -extent_one

        extent_two = 0.0
        if param_dict['endBound'] not in ['BLIND', 'SYMMETRIC']:
            raise NotImplementedError("endBound type not supported: {}".format(param_dict['endBound']))

        elif 'hasSecondDirection' in param_dict and param_dict['hasSecondDirection'] is True:
            if param_dict['secondDirectionBound'] != 'BLIND':
                raise NotImplementedError("secondDirectionBound type not supported: {}".format(param_dict['endBound']))
            extent_type = 'TwoSidesFeatureExtentType'
            extent_two = self._expr2meter(param_dict['secondDirectionDepth'])
            if 'secondDirectionOppositeDirection' in param_dict \
                and str(param_dict['secondDirectionOppositeDirection']) == 'true':
                extent_two = -extent_two

        else:
            extent_type = EXTENT_TYPE_MAP[param_dict['endBound']]

        operation = OPERATION_MAP[param_dict['operationType']]

        save_dict = {"name": feature_data['name'],
                    "type": "ExtrudeFeature",
                    "profiles": profiles,
                    "operation": operation,
                    "start_extent": {"type": "ProfilePlaneStartDefinition"},
                    "extent_type": extent_type,
                    "extent_one": {
                        "distance": {
                            "type": "ModelParameter",
                            "value": extent_one,
                            "name": "none",
                            "role": "AlongDistance"
                        },
                        "taper_angle": {
                            "type": "ModelParameter",
                            "value": 0.0,
                            "name": "none",
                            "role": "TaperAngle"
                        },
                        "type": "DistanceExtentDefinition"
                    },
                    "extent_two": {
                        "distance": {
                            "type": "ModelParameter",
                            "value": extent_two,
                            "name": "none",
                            "role": "AgainstDistance"
                        },
                        "taper_angle": {
                            "type": "ModelParameter",
                            "value": 0.0,
                            "name": "none",
                            "role": "Side2TaperAngle"
                        },
                        "type": "DistanceExtentDefinition"
                    },
                    }
        return save_dict

    def _parse_revolve(self, feature_data):
        """
        解析旋转参数
        :param feature_data:
        :return:
        """
        save_dict = {}
        return save_dict

    def _parse_loft(self, feature_data):
        """
        解析旋转参数
        :param feature_data:
        :return:
        """
        save_dict = {}
        return save_dict

    def _parse_sweep(self, feature_data):
        """
        解析旋转参数
        :param feature_data:
        :return:
        """
        save_dict = {}
        return save_dict

    def _parse_bounding_box(self):
        bbox_info = self.client.eval_bounding_box(self.did, self.wid, self.eid)
        result = {"type": "BoundingBox3D",
                  "max_point": xyz_list2dict(bbox_info['maxCorner']),
                  "min_point": xyz_list2dict(bbox_info['minCorner'])}
        return result

    def parse(self):
        """
        parse into fusion360 gallery format,
        only sketch and extrusion are supported.
        """
        result = {"entities": OrderedDict(), "properties": {}, "sequence": []}

        # bbox = self._parse_bounding_box()
        # result["properties"].update({"bounding_box": bbox})

        for i, feat_item in enumerate(self.feature_list['features']):
            feat_data = feat_item['message']
            feat_type = feat_data['featureType']
            feat_id = feat_data['featureId']

            if feat_type == 'newSketch':
                feat_dict = self._parse_sketch(feat_data)
                for pf_key in feat_dict['profiles'].keys():
                    self.profile2sketch[pf_key] = feat_id

            elif feat_type == 'extrude':
                feat_dict = self._parse_extrude(feat_data)

            elif feat_type == 'revolve':  # 旋转
                feat_dict = self._parse_revolve(feat_data)

            elif feat_type == 'loft':  # 放样
                feat_dict = self._parse_loft(feat_data)

            elif feat_type == 'sweep':  # 扫描
                feat_dict = self._parse_sweep(feat_data)

            elif feat_type == '线性阵列、圆周阵列？':
                feat_dict = self._parse_sweep(feat_data)

            else:
                raise NotImplementedError("unsupported feature type: {}".format(feat_type))

            result["entities"][feat_id] = feat_dict
            result["sequence"].append({"index": i, "type": feat_dict['type'], "entity": feat_id})

        # with open(r'E:\document\DeeplearningIdea\multi_cmd_seq_gen\all_sketches.json', 'w') as f:
        #     json.dump(feat_dict, f, ensure_ascii=False, indent=4)
        # exit('---------sketch info save finished-------')

        return result


class ExtrudeParser(object):
    """
    解析拉伸操作
    """
    def __init__(self, fea_item_ofs):
        self.fea = fea_item_ofs

    def _parse_extrude(self, fea_item_ofs):
        """
        解析拉伸参数
        :param fea_item_ofs:
        :return:
        """
        param_dict = self.parse_feature_param(fea_item_ofs['message']['parameters'])
        if 'hasOffset' in param_dict and param_dict['hasOffset'] is True:
            raise NotImplementedError("extrude with offset not supported: {}".format(param_dict['hasOffset']))

        entities = param_dict['entities']  # geometryIds for target face
        profiles = self._locate_sketch_profile(entities)  # JGC

        extent_one = self._expr2meter(param_dict['depth'])
        if param_dict['endBound'] == 'SYMMETRIC':
            extent_one = extent_one / 2

        if 'oppositeDirection' in param_dict and param_dict['oppositeDirection'] is True:
            extent_one = -extent_one

        extent_two = 0.0
        if param_dict['endBound'] not in ['BLIND', 'SYMMETRIC']:
            raise NotImplementedError("endBound type not supported: {}".format(param_dict['endBound']))

        elif 'hasSecondDirection' in param_dict and param_dict['hasSecondDirection'] is True:
            if param_dict['secondDirectionBound'] != 'BLIND':
                raise NotImplementedError("secondDirectionBound type not supported: {}".format(param_dict['endBound']))
            extent_type = 'TwoSidesFeatureExtentType'
            extent_two = self._expr2meter(param_dict['secondDirectionDepth'])
            if 'secondDirectionOppositeDirection' in param_dict \
                and str(param_dict['secondDirectionOppositeDirection']) == 'true':
                extent_two = -extent_two

        else:
            extent_type = EXTENT_TYPE_MAP[param_dict['endBound']]

        operation = OPERATION_MAP[param_dict['operationType']]

        save_dict = {"name": feature_data['name'],
                    "type": "ExtrudeFeature",
                    "profiles": profiles,
                    "operation": operation,
                    "start_extent": {"type": "ProfilePlaneStartDefinition"},
                    "extent_type": extent_type,
                    "extent_one": {
                        "distance": {
                            "type": "ModelParameter",
                            "value": extent_one,
                            "name": "none",
                            "role": "AlongDistance"
                        },
                        "taper_angle": {
                            "type": "ModelParameter",
                            "value": 0.0,
                            "name": "none",
                            "role": "TaperAngle"
                        },
                        "type": "DistanceExtentDefinition"
                    },
                    "extent_two": {
                        "distance": {
                            "type": "ModelParameter",
                            "value": extent_two,
                            "name": "none",
                            "role": "AgainstDistance"
                        },
                        "taper_angle": {
                            "type": "ModelParameter",
                            "value": 0.0,
                            "name": "none",
                            "role": "Side2TaperAngle"
                        },
                        "type": "DistanceExtentDefinition"
                    },
                    }
        return save_dict

    @staticmethod
    def parse_feature_param(fea_item_msg_param_ofs):
        param_dict = {}
        for i, param_item in enumerate(fea_item_msg_param_ofs):
            param_msg = param_item['message']
            param_id = param_msg['parameterId']

            if 'queries' in param_msg:
                param_value = []
                for j in range(len(param_msg['queries'])):
                    param_value.extend(param_msg['queries'][j]['message']['geometryIds'])

            elif 'expression' in param_msg:
                param_value = param_msg['expression']

            elif 'value' in param_msg:
                param_value = param_msg['value']

            else:
                raise NotImplementedError('param_msg:\n{}'.format(param_msg))

            param_dict[param_id] = param_value
        return param_dict

    def _locate_sketch_profile(self, geo_ids):
        return [{"profile": k, "sketch": self.profile2sketch[k]} for k in geo_ids]

    def _expr2meter(self, expr):
        return self.client.expr2meter(self.did, self.wid, self.eid, expr)



