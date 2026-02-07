"""
解析拉伸等建模命令参数
"""
from colorama import Fore, Style


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
            print(Fore.RED + f'not considered operation paramId: {param_id}, save directively')
            # raise NotImplementedError('param_msg:\n{}'.format(param_msg))

        param_dict[param_id] = param_value
    return param_dict


class Extrude(object):
    """
    拉伸命令
    目前仅支持: [单侧拉伸, 两侧对称拉伸], 不支持两侧分别进行距离不等的拉伸,拉伸到实体\曲面\顶点
    """
    def __init__(self, operation_type, region, depth, direction, draft_angle, is_symmetric, is_opposite_dir):
        """
        拉伸命令参数
        注意: 如果是拉伸切除，那么默认方向为所在面法向的反向。但是如果指定了拉伸方向，则还是按指定轴方向来
        :param operation_type: 操作方式，["NEW", "REMOVE", "ADD", "INTERSECT"]
        :param region: 拉伸的区域 id，数组
        :param depth: 总拉伸深度
        :param direction: 拉伸方向，直线实体 id，数组，使用默认法线方向为 None
        :param draft_angle: 拔模斜度，不使用时为 None
        :param is_symmetric: 是否对称拉伸
        :param is_opposite_dir: 是否拉伸方向反向
        """
        self.region = region
        self.depth = depth
        self.direction = direction
        self.operation_type = operation_type
        self.draft_angle = draft_angle
        self.is_symmetric = is_symmetric
        self.is_opposite_dir = is_opposite_dir

        # 所需的全部实体 id
        self.required_geo = self.region if self.direction is None else self.region.extend(self.direction)

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取拉伸参数

        可获得的参数如下:

        key = feat_item_msg_param_item["message"]["parameterId"]

        domain: "MODEL"
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
            direction = param_dict['extrudeDirection']
        else:
            direction = None

        operation_type = param_dict['operationType']

        # 拔模斜度
        if param_dict['hasDraft']:
            draft_angle = param_dict['draftAngle']
        else:
            draft_angle = None

        # 是否两侧对称拉伸
        is_symmetric = param_dict['symmetric']

        # 拉伸方向是否与面法向反向
        is_opposite_dir = param_dict['oppositeDirection']

        return cls(operation_type, region, depth, direction, draft_angle, is_symmetric, is_opposite_dir)


class Revolve(object):
    """
    仅支持指定角度的旋转, 不支持旋转到指定实体
    """
    def __init__(self, operation_type, region, axis, angle, is_symmetric, is_opposite_dir):
        self.operation_type = operation_type  # 拉伸还是拉伸切除
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
        - operationType: "ADD"
        surfaceOperationType: "NEW". 未知
        - entities: ["queries"][0]["message"]["geometryIds"] = ["JJC"]. 旋转涉及的区域 ID
        surfaceEntities: . 未知
        wallShape: . 未知
        midplane: false. 未知
        thickness1: "0.25 in". 未知
        flipWall: false. 未知
        thickness2: "0 in". 未知
        thickness: "0.25 in". 未知
        - axis: ["queries"][0]["message"]["geometryIds"] = ["JHF"]. 旋转轴涉及的区域 ID
        - fullRevolve: false. 是否整周旋转
        endBound: "BLIND". 结束面类型, "BLIND" 为指定角度的旋转
        - oppositeDirection: false. 是否沿法线反向旋转
        - angle: "120 deg". 旋转角度
        - symmetric: false. 是否对称旋转
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

        operation_type = param_dict['operationType']
        region = param_dict['entities']  # 引用的几何体，例如 JDC，具体定义后续统一获取

        axis = param_dict['axis']

        if param_dict['fullRevolve']:
            angle = '360 deg'
        else:
            angle = param_dict['angle']

        is_symmetric = param_dict['symmetric']
        is_opposite_dir = param_dict['oppositeDirection']

        return cls(operation_type, region, axis, angle, is_symmetric, is_opposite_dir)


class Sweep(object):
    """
    仅支持指定路径的扫描
    """
    def __init__(self, operation_type, region, path):
        self.operation_type = operation_type
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
        - operationType: "ADD".
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

        operation_type = param_dict['operationType']
        region = param_dict['profiles']
        path = param_dict['path']

        return cls(operation_type, region, path)


class Loft(object):
    """
    只支持几个轮廓，不支持带引导线的放样
    """

    def __init__(self, operation_type, sections):
        self.operation_type = operation_type
        self.sections = sections

        # 所需的全部实体 id
        self.required_geo = self.sections

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        bodyType: "SOLID".
        - operationType: "ADD".
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

        operation_type = param_dict['operationType']
        sections = param_dict['sheetProfilesArray']

        return cls(operation_type, sections)


class Fillet(object):
    """
    圆角
    """
    def __init__(self, entities, radius):
        """
        圆角对象可能是面，也可能是边
        """
        self.entities = entities
        self.radius = radius

        # 所需的全部实体 id
        self.required_geo = self.entities

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取扫描参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        filletType: "EDGE"
        - entities: ["queries"][0]["message"]["geometryIds"] = ["JHS"].
        side1Face:
        side2Face:
        centerFaces:
        tangentPropagation: true.
        blendControlType: "RADIUS".
        crossSection: "CIRCULAR".
        - radius: "0.1 in", "isInteger": false. 圆角半径
        width: "0.2 in", "isInteger": false.
        rho: "0.5", "isInteger": false.
        magnitude: "0.5", "isInteger": false.
        defaultsChanged: true.
        isAsymmetric: false.
        otherRadius: "0.2 in", "isInteger": false.
        flipAsymmetric: false.
        isPartial: false.
        startPartialType: "PERCENTAGE".
        startPartialOffset: "0.2 in", "isInteger": false.
        startPartialEntity: .
        partialFirstEdgeTotalParameter: "0.01", "isInteger": false.
        partialOppositeParameter: true.
        useTrimmedFirstBound: false.
        secondBound: false.
        endPartialType: "PERCENTAGE".
        endPartialOffset: "0.2 in", "isInteger": false.
        endPartialEntity: .
        useTrimmedSecondBound: false.
        partialSecondEdgeTotalParameter: "0.99", "isInteger": false.
        isVariable: false.
        vertexSettings: .
        pointOnEdgeSettings: .
        smoothTransition: false.
        allowEdgeOverflow: true.
        keepEdges: .
        smoothCorners: false.
        smoothCornerExceptions: .

        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])

        entities = param_dict['entities']
        radius = param_dict['radius']

        return cls(entities, radius)


class Chamfer(object):
    """
    倒斜角，仅支持两边相等的倒斜角
    """
    def __init__(self, entities, width):
        self.entities = entities
        self.width = width

        # 所需的全部实体 id
        self.required_geo = self.entities

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取扫描参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        - entities: ["queries"][0]["message"]["geometryIds"] = ["JLi"].
        chamferMethod: "FACE_OFFSET".
        chamferType: "EQUAL_OFFSETS".
        - width: "0.05 in", "isInteger": false. 倒斜角等腰直角三角形腰长
        width1: "0.2 in", "isInteger": false.
        oppositeDirection: false.
        width2: "0.2 in", "isInteger": false.
        angle: "45 deg", "isInteger": false.
        directionOverrides: .
        tangentPropagation: true.

        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])

        entities = param_dict['entities']
        width = param_dict['width']

        return cls(entities, width)


class LinearPattern(object):
    """
    线性阵列，最多支持两个方向，不支持跳过实例
    仅支持实体阵列，不支持特征阵列
    TODO: 基于特征的阵列
    """
    def __init__(self,
                 operation_type,
                 entities,
                 direction_one,
                 distance,
                 instance_count,
                 opposite_direction,
                 direction_two,
                 distance_two,
                 instance_count_two,
                 opposite_direction_two
                 ):
        self.operation_type = operation_type
        self.entities = entities

        self.direction_one = direction_one
        self.distance = distance
        self.instance_count = instance_count
        self.opposite_direction = opposite_direction

        self.direction_two = direction_two
        self.distance_two = distance_two
        self.instance_count_two = instance_count_two
        self.opposite_direction_two = opposite_direction_two

        # 所需的全部实体 id
        self.required_geo = self.entities + self.direction_one if self.direction_two is None else self.entities + self.direction_one + self.direction_two

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取扫描参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        patternType: "PART". 实体阵列，也可能是 "FEATURE"，"FACE"。
        - operationType: "ADD".
        - entities: ["queries"][0]["message"]["geometryIds"] = ["JHD"].
        faces: .
        instanceFunction: "featureIds" = ["FpT0Xmz3iQFCaSQ_1"]. 表示特征的 id
        - directionOne: ["queries"][0]["message"]["geometryIds"] = ["JI5"].
        - distance: "1 in", "isInteger": false. 相邻实例之间的间距
        - instanceCount: "2", "isInteger": true. 阵列实例数（包含自身）
        - oppositeDirection: false. 是否与所选实体反向.
        isCentered: false.
        - hasSecondDir: false.
        - directionTwo: .
        - distanceTwo: .
        - instanceCountTwo: .
        - oppositeDirectionTwo: false.
        isCenteredTwo: false.
        defaultScope: false.
        booleanScope: .
        fullFeaturePattern: false.
        skipInstances: false.
        skippedInstances: [].

        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])

        operation_type = param_dict['operationType']
        entities = param_dict['entities']

        direction_one = param_dict['directionOne']
        distance = param_dict['distance']
        instance_count = param_dict['instanceCount']
        opposite_direction = param_dict['oppositeDirection']

        has_second_dir = param_dict['hasSecondDir']
        if has_second_dir:
            direction_two = param_dict['directionTwo']
            distance_two = param_dict['distanceTwo']
            instance_count_two = param_dict['instanceCountTwo']
            opposite_direction_two = param_dict['oppositeDirectionTwo']

        else:
            direction_two = None
            distance_two = None
            instance_count_two = None
            opposite_direction_two = None

        return cls(operation_type, entities,
                   direction_one, distance, instance_count, opposite_direction,
                   direction_two, distance_two, instance_count_two, opposite_direction_two)


class CircularPattern(object):
    """
    圆周阵列，仅支持等间距，不支持跳过实例
    TODO: 基于特征的阵列
    """
    def __init__(self, operation_type, entities, axis, angle, instance_count, opposite_direction):
        self.operation_type = operation_type
        self.entities = entities
        self.axis = axis
        self.angle = angle
        self.instance_count = instance_count
        self.opposite_direction = opposite_direction

        # 所需的全部实体 id
        self.required_geo = self.entities + self.axis

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取扫描参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        patternType: "PART".
        - operationType: "ADD".
        - entities: ["queries"][0]["message"]["geometryIds"] = ["JHD"].
        faces: .
        instanceFunction: .
        - axis: ["queries"][0]["message"]["geometryIds"] = ["KZGC"]. 旋转轴
        - angle: "360 deg", "isInteger": false.
        - instanceCount: "3", "isInteger": true.
        - oppositeDirection: "false".
        equalSpace: true.
        isCentered: false.
        defaultScope: false.
        booleanScope: .
        fullFeaturePattern: false.
        skipInstances: false.
        skippedInstances: [].

        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])
        operation_type = param_dict['operationType']
        entities = param_dict['entities']

        axis = param_dict['axis']
        angle = param_dict['angle']
        instance_count = param_dict['instanceCount']
        opposite_direction = param_dict['oppositeDirection']

        return cls(operation_type, entities, axis, angle, instance_count, opposite_direction)


class Draft(object):
    """
    拔模
    """
    def __init__(self, draft_faces, along_pull: bool, angle):
        self.draft_faces = draft_faces
        self.along_pull = along_pull
        self.angle = angle

        # 所需的全部实体 id
        self.required_geo = self.draft_faces

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取扫描参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        draftFeatureType: "NEUTRAL_PLANE".
        neutralPlane: ["queries"][0]["message"]["geometryIds"] = ["KZKC"].
        - draftFaces: ["queries"][0]["message"]["geometryIds"] = ["KZGC"].  拔模面
        pullDirectionEntity: .
        partingEdges: .
        hintFaces: .
        partingLineSides: "ONE_SIDED".
        - alongPull: true. 上小下大，拔模面大小不变
        - angle: "3 deg", "isInteger": false. 拔模斜度
        pullDirection: false.
        secondAngle: "3 deg", "isInteger": false.
        secondPullDirection: true.
        tangentPropagation: true.
        referenceEdgePropagation: true.
        reFillet: false.

        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])

        draft_faces = param_dict['draftFaces']
        along_pull = param_dict['alongPull']
        angle = param_dict['angle']

        return cls(draft_faces, along_pull, angle)


class Rib(object):
    """
    筋
    """
    def __init__(self, profiles, parts, thickness, rib_extrusion_direction, opposite_direction):
        self.profiles = profiles
        self.parts = parts
        self.thickness = thickness
        self.rib_extrusion_direction = rib_extrusion_direction
        self.opposite_direction = opposite_direction

        # 所需的全部实体 id
        self.required_geo = self.profiles + self.parts

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取扫描参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        - profiles: ["queries"][0]["message"]["geometryIds"] = ["JiB"]. 筋对应的草图
        - parts: 筋需要成型到的实体
        - thickness: 筋厚度
        - ribExtrusionDirection: "PARALLEL_TO_SKETCH_PLANE". 筋成型方向
        - oppositeDirection: false. 是否反向
        hasDraft: false. 是否有拔模斜度
        draftAngle: "3 deg", "isInteger": false.
        draftOpposite: "false".
        extendProfilesUpToPart: false.
        mergeRibs: true.

        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])

        profiles = param_dict['profiles']
        parts = param_dict['parts']
        thickness = param_dict['thickness']
        rib_extrusion_direction = param_dict['ribExtrusionDirection']
        opposite_direction = param_dict['oppositeDirection']

        return cls(profiles, parts, thickness, rib_extrusion_direction, opposite_direction)


class Mirror(object):
    """
    镜像
    """
    def __init__(self, operation_type, entities, mirror_plane):
        self.operation_type = operation_type
        self.entities = entities
        self.mirror_plane = mirror_plane

        # 所需的全部实体 id
        self.required_geo = self.entities + self.mirror_plane

    @classmethod
    def from_ofs(cls, feat_item_ofs):
        """
        从指定特征的列表获取扫描参数

        可获得的参数如下:
        key = feat_item_msg_param_item["message"]["parameterId"]

        patternType: "PART".  还有 "FEATURE", "FACE"
        - operationType: "ADD".
        - entities: ["queries"][0]["message"]["geometryIds"] = ["JHD"].
        faces: .
        instanceFunction: .
        - mirrorPlane: ["queries"][0]["message"]["geometryIds"] = ["KhOB"].
        defaultScope: false.
        booleanScope: ["queries"][0]["message"]["geometryIds"] = ["JHD"].
        fullFeaturePattern: false.

        """
        param_dict = parse_feature_param(feat_item_ofs['message']['parameters'])

        operation_type = param_dict['operationType']
        entities = param_dict['entities']
        mirror_plane = param_dict['mirrorPlane']

        return cls(operation_type, entities, mirror_plane)


