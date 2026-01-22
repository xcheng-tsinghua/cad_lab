from collections import OrderedDict


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



