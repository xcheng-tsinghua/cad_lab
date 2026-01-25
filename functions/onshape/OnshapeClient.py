import json
import string
import random
import mimetypes
from functions.onshape.OnshapeAPI import OnshapeAPI
import os


class Client(object):
    """
    Defines methods for testing the Onshape API. Comes with several methods:

    - Create a document
    - Delete a document
    - Get a list of documents

    Attributes:
        - stack (str, default='https://cad.onshape.com'): Base URL
        - logging (bool, default=True): Turn logging on or off
    """
    def __init__(self, creds='./config/onshape_credit.json'):
        """
        Instantiates a new Onshape client.

        Args:
            - stack (str, default='https://cad.onshape.com'): Base URL
            - logging (bool, default=True): Turn logging on or off
        """
        self._api = OnshapeAPI(creds=creds)

    def new_document(self, name='Test Document', owner_type=0, public=False):
        """
        Create a new document.

        Args:
            - name (str, default='Test Document'): The doc name
            - owner_type (int, default=0): 0 for user, 1 for company, 2 for team
            - public (bool, default=False): Whether or not to make doc public

        Returns:
            - requests.Response: Onshape response data
        """
        payload = {
            'name': name,
            'ownerType': owner_type,
            'isPublic': public
        }

        return self._api.request('post', '/api/documents', body=payload)

    def rename_document(self, did, name):
        """
        Renames the specified document.

        Args:
            - did (str): Document ID
            - name (str): New document name

        Returns:
            - requests.Response: Onshape response data
        """
        payload = {
            'name': name
        }

        return self._api.request('post', '/api/documents/' + did, body=payload)

    def del_document(self, did):
        """
        Delete the specified document.

        Args:
            - did (str): Document ID

        Returns:
            - requests.Response: Onshape response data
        """
        return self._api.request('delete', '/api/documents/' + did)

    def get_document(self, did):
        """
        Get details for a specified document.

        Args:
            - did (str): Document ID

        Returns:
            - requests.Response: Onshape response data
        """
        return self._api.request('get', '/api/documents/' + did)

    def list_documents(self):
        """
        Get list of documents for current user.

        Returns:
            - requests.Response: Onshape response data
        """
        return self._api.request('get', '/api/documents')

    def create_assembly(self, did, wid, name='My Assembly'):
        """
        Creates a new assembly element in the specified document / workspace.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - name (str, default='My Assembly')

        Returns:
            - requests.Response: Onshape response data
        """
        payload = {
            'name': name
        }

        return self._api.request('post', '/api/assemblies/d/' + did + '/w/' + wid, body=payload)

    def request_features(self, model_url):
        """
        Gets the feature list for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        """
        v_list = model_url.split("/")
        did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

        res = self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features')
        return res.json()

    def get_partstudio_tessellatededges(self, did, wid, eid):
        """
        Gets the tessellation of the edges of all parts in a part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        """
        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/tessellatededges')

    def upload_blob(self, did, wid, filepath='./blob.json'):
        """
        Uploads a file to a new blob element in the specified doc.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - filepath (str, default='./blob.json'): Blob element location

        Returns:
            - requests.Response: Onshape response data
        """
        chars = string.ascii_letters + string.digits
        boundary_key = ''.join(random.choice(chars) for _ in range(8))

        mimetype = mimetypes.guess_type(filepath)[0]
        encoded_filename = os.path.basename(filepath)
        file_content_length = str(os.path.getsize(filepath))
        blob = open(filepath)

        req_headers = {
            'Content-Type': 'multipart/form-data; boundary="%s"' % boundary_key
        }

        # build request body
        payload = '--' + boundary_key + '\r\nContent-Disposition: form-data; name="encodedFilename"\r\n\r\n' + encoded_filename + '\r\n'
        payload += '--' + boundary_key + '\r\nContent-Disposition: form-data; name="fileContentLength"\r\n\r\n' + file_content_length + '\r\n'
        payload += '--' + boundary_key + '\r\nContent-Disposition: form-data; name="file"; filename="' + encoded_filename + '"\r\n'
        payload += 'Content-Type: ' + mimetype + '\r\n\r\n'
        payload += blob.read()
        payload += '\r\n--' + boundary_key + '--'

        return self._api.request('post', '/api/blobelements/d/' + did + '/w/' + wid, headers=req_headers, body=payload)

    def part_studio_stl(self, did, wid, eid):
        """
        Exports STL export from a part studio

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        """
        req_headers = {
            'Accept': 'application/vnd.onshape.v1+octet-stream'
        }
        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/stl', headers=req_headers)


class OnshapeClient(Client):
    """
    inherited from OnShape public apikey python client,
    with additional method for parsing cad.
    """
    def get_tessellatedfaces(self, did, wid, eid):
        """
        Gets the feature list for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: OnShape response data
        """
        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/tessellatedfaces')

    def get_entity_by_id(self, did, wid, eid, geo_id, entity_type):
        """
        get the parameters of geometry entity for specified entity id and type

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - geo_id (str): geometry entity ID
            - entity_type (str): 'VERTEX', 'EDGE' or 'FACE'

        Returns:
            - requests.Response: OnShape response data
        """
        func_dict = {"VERTEX": ("evVertexPoint", "vertex"),
                     "EDGE": ("evCurveDefinition", "edge"),
                     "FACE": ("evSurfaceDefinition", "face")}
        body = {
            "script":
                "function(context is Context, queries) { " +
                "   var res_list = [];"
                "   var q_arr = evaluateQuery(context, queries.id);"
                "   for (var i = 0; i < size(q_arr); i+= 1){"
                "       var res = %s(context, {\"%s\": q_arr[i]});" % (func_dict[entity_type][0], func_dict[entity_type][1]) +
                "       res_list = append(res_list, res);"
                "   }"
                "   return res_list;"
                "}",
            "queries": [{"key": "id", "value": geo_id}]
        }
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        return res

    def get_face_by_id(self, did, wid, eid, geo_id):
        """
        注意 geo_id 需要以数组的形式输入
        一个 id 的情况: geo_id = [id]
        多个 id 的情况: geo_id = [id1, id2, ...]

        get the parameters of geometry entity for specified entity id and type

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - geo_id (str): geometry entity ID
            - entity_type (str): 'VERTEX', 'EDGE' or 'FACE'

        Returns:
            - requests.Response: OnShape response data
        """
        body = {
            "script":
                "function(context is Context, queries) { "
                "   var res_list = [];"
                "   var q_arr = evaluateQuery(context, queries.id);"
                "   for (var i = 0; i < size(q_arr); i+= 1){"
                "       var res = evSurfaceDefinition(context, {\"face\": q_arr[i]});"
                "       res_list = append(res_list, res);"
                "   }"
                "   return res_list;"
                "}",
            "queries": [{"key": "id", "value": geo_id}]
        }
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        return res

    def request_multi_feat_topology(self, model_url, fea_id_list):
        """
        通过草图特征或者拉伸等特征的 id 解析拓扑结构，包含草图区域，边、角点等

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - feat_id (str): Feature ID of a sketch or operation

        Returns:
            - dict: a hierarchical parametric representation
        """

        body = {
            "script":
                "function(context is Context, queries) { "
                "   var res_list = [];"
                "   var q_arr = [" + ",".join([f"\"{fid}\"" for fid in fea_id_list]) + "];"
                "   for (var l = 0; l < size(q_arr); l+= 1){"
                "       var topo = {};"
                # "       topo.regions = [];"
                "       topo.faces = [];"
                "       topo.edges = [];"
                "       topo.vertices = [];"

                # "       /* ---------- 1. Regions (regions only) ---------- */"  # 区域列表，每个区域仅包含：区域的定义、区域id、该区域下的边id。但实测 region 不如 face 获得的信息多
                # "       var q_region = qSketchRegion(makeId(q_arr[l]));"
                # "       var region_arr = evaluateQuery(context, q_region);"
                # "       for (var i = 0; i < size(region_arr); i += 1) {"
                # "           var region_topo = {};"
                # "           region_topo.id = transientQueriesToStrings(region_arr[i]);"  # 区域id
                # "           region_topo.edges = [];"  # 该区域下的边id
                # "           region_topo.param = evSurfaceDefinition(context, {face: region_arr[i]});"  # 区域的定义
                # "           var q_edge = qAdjacent(region_arr[i], AdjacencyType.EDGE, EntityType.EDGE);"
                # "           var edge_arr = evaluateQuery(context, q_edge);"
                # "           for (var j = 0; j < size(edge_arr); j += 1) {"
                # "               const edge_id = transientQueriesToStrings(edge_arr[j]);"
                # "               region_topo.edges = append(region_topo.edges, edge_id);"
                # "           }"
                # "           topo.regions = append(topo.regions, region_topo);"
                # "       }"
                                   
                "      /* ---------- 2. Face (ALL faces generated) ---------- */"  # 面列表，每个面仅包含：面的定义、面id、该面下的边id。
                "       var q_face = qCreatedBy(makeId(q_arr[l]), EntityType.FACE);"
                "       var face_arr = evaluateQuery(context, q_face);"
                "       for (var i = 0; i < size(face_arr); i += 1) {"
                "           var face_topo = {};"
                "           face_topo.id = transientQueriesToStrings(face_arr[i]);"  # 面id
                "           face_topo.edges = [];"  # 该面下的边id
                "           face_topo.param = evSurfaceDefinition(context, {face: face_arr[i]});"  # 面的定义
                "           var q_edge = qAdjacent(face_arr[i], AdjacencyType.EDGE, EntityType.EDGE);"
                "           var edge_arr = evaluateQuery(context, q_edge);"
                "           for (var j = 0; j < size(edge_arr); j += 1) {"
                "               const edge_id = transientQueriesToStrings(edge_arr[j]);"
                "               face_topo.edges = append(face_topo.edges, edge_id);"
                "           }"
                "           topo.faces = append(topo.faces, face_topo);"
                "       }"
                                                                                        
                "      /* ---------- 3. Edges (ALL sketch edges, open or closed) ---------- */"  # 边列表，每个边仅包含：边的定义、边的id、该边下的端点id
                "      var q_edge = qCreatedBy(makeId(q_arr[l]), EntityType.EDGE);"
                "      var edge_arr = evaluateQuery(context, q_edge);"
                "      for (var j = 0; j < size(edge_arr); j += 1) {"
                "          var edge_topo = {};"
                "          const edge_id = transientQueriesToStrings(edge_arr[j]);"
                "          edge_topo.id = edge_id;"  # 边的id
                "          edge_topo.vertices = [];"  # 该边下的端点id
                "          edge_topo.param = evCurveDefinition(context, {edge: edge_arr[j]});"  # 边的定义
                "          var q_vertex = qAdjacent(edge_arr[j], AdjacencyType.VERTEX, EntityType.VERTEX);"
                "          var vertex_arr = evaluateQuery(context, q_vertex);"
                "          for (var k = 0; k < size(vertex_arr); k += 1) {"
                "              const vertex_id = transientQueriesToStrings(vertex_arr[k]);"
                "              edge_topo.vertices = append(edge_topo.vertices, vertex_id);"
                "          }"
                "          topo.edges = append(topo.edges, edge_topo);"                                                           
                "      }"
                                                                                        
                "       /* ---------- 4. Vertices (ALL sketch vertices) ---------- */"  # 点列表，每个点仅包含：点的定义、点的id
                "      var q_vertex = qCreatedBy(makeId(q_arr[l]), EntityType.VERTEX);"
                "      var vertex_arr = evaluateQuery(context, q_vertex);"
                "      for (var k = 0; k < size(vertex_arr); k += 1) {"
                "          var vertex_topo = {};"
                "          const vertex_id = transientQueriesToStrings(vertex_arr[k]);"
                "          vertex_topo.id = vertex_id;"  # 点的id
                "          vertex_topo.param = evVertexPoint(context, {vertex: vertex_arr[k]});"  # 点的定义
                "          topo.vertices = append(topo.vertices, vertex_topo);"
                "      }"
                "      res_list = append(res_list, topo);"
                "   }"
                "   return res_list;"
                "}",
            "queries": []
        }

        v_list = model_url.split("/")
        did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

        res = self._api.request('post',
                                '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript',
                                body=body)

        return res.json()

    def request_multi_entity_topology(self, model_url, ent_id_list):
        """
        通过草图特征或者拉伸等特征的 id 解析拓扑结构，包含草图区域，边、角点等

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - feat_id (str): Feature ID of a sketch or operation

        Returns:
            - dict: a hierarchical parametric representation
        """
        body = {
            'script': '''function(context is Context, queries) {
            // var all_eval_ids = []; // 已评估的id，不会重复评估
            var res_list = [];  // 最终返回的结果列表

            var q_entity_list = evaluateQuery(context, queries.id);
            for (var i = 0; i < size(q_entity_list); i+= 1){ // entityId 食欲个包含了诸如 JDC、JGI的id列表
                var q_entity = q_entity_list[i];

                const entity_id = transientQueriesToStrings(q_entity);

                // 判断是否已评估该id对应的实体
                // if (isIn(entity_id, all_eval_ids)){continue;}
                // all_eval_ids = append(all_eval_ids, entity_id);

                var topo = {};  // 设置当前id对应的属性
                topo.id = entity_id;
                topo.faces = [];  // 如果当前id是面，那么face数组size为1，edge数组包含该面下的边，vertices数组包含所有边下的所有点
                topo.edges = [];
                topo.vertices = [];

                var isBody = size(evaluateQuery(context, qEntityFilter(q_entity, EntityType.BODY))) == 1;
                var isFace = size(evaluateQuery(context, qEntityFilter(q_entity, EntityType.FACE))) == 1;
                var isEdge = size(evaluateQuery(context, qEntityFilter(q_entity, EntityType.EDGE))) == 1;
                var isVertex = size(evaluateQuery(context, qEntityFilter(q_entity, EntityType.VERTEX))) == 1;

                if (isFace)
                {
                    // face 需要获取其id、定义、边列表
                    const face_id = transientQueriesToStrings(q_entity);

                    // 判断是否已评估该id对应的实体
                    // if (isIn(face_id, all_eval_ids)){continue;}
                    // all_eval_ids = append(all_eval_ids, face_id);

                    // 设置主拓扑类型
                    topo.entityType = "FACE";

                    // 获取face的id、定义、其下的边
                    var face_topo = {};
                    face_topo.id = face_id;
                    face_topo.param = evSurfaceDefinition(context, {face: q_entity});
                    face_topo.edges = [];

                    var q_edges = evaluateQuery(context, qAdjacent(q_entity, AdjacencyType.EDGE, EntityType.EDGE));
                    for (var j = 0; j < size(q_edges); j += 1) {
                        // 每个边仅包含：边的定义、边的id、该边下的端点id
                        var q_edge = q_edges[j];
                        const edge_id = transientQueriesToStrings(q_edge);
                        face_topo.edges = append(face_topo.edges, edge_id);

                        // 判断是否已评估该id对应的实体
                        // if (isIn(edge_id, all_eval_ids)){continue;}
                        // all_eval_ids = append(all_eval_ids, edge_id);

                        var edge_topo = {};
                        edge_topo.id = edge_id;
                        edge_topo.param = evCurveDefinition(context, {edge: q_edge});
                        edge_topo.vertices = [];

                        var q_vertices = evaluateQuery(context, qAdjacent(q_edge, AdjacencyType.VERTEX, EntityType.VERTEX));
                        for (var k = 0; k < size(q_vertices); k += 1) {
                            var q_vertex = q_vertices[k];
                            const vertex_id = transientQueriesToStrings(q_vertex);
                            edge_topo.vertices = append(edge_topo.vertices, vertex_id);

                            // 判断是否已评估该id对应的实体
                            // if (isIn(vertex_id, all_eval_ids)){continue;}
                            // all_eval_ids = append(all_eval_ids, vertex_id);

                            // 每个点仅包含：点的定义、点的id
                            var vertex_topo = {};

                            // 评估每个角点的属性
                            vertex_topo.id = vertex_id;
                            vertex_topo.param = evVertexPoint(context, {vertex: q_vertex});

                            topo.vertices = append(topo.vertices, vertex_topo);
                        }

                        topo.edges = append(topo.edges, edge_topo);
                    }

                    topo.faces = append(topo.faces, face_topo);
                }

                else if (isEdge)
                {
                    // 每个边仅包含：边的定义、边的id、该边下的端点id
                    const edge_id = transientQueriesToStrings(q_entity);

                    // 判断是否已评估该id对应的实体
                    // if (isIn(edge_id, all_eval_ids)){continue;}
                    // all_eval_ids = append(all_eval_ids, edge_id);

                    // 设置主拓扑类型
                    topo.entityType = "EDGE";

                    var edge_topo = {};
                    edge_topo.id = edge_id;
                    edge_topo.param = evCurveDefinition(context, {edge: q_entity});
                    edge_topo.vertices = [];

                    var q_vertices = evaluateQuery(context, qAdjacent(q_entity, AdjacencyType.VERTEX, EntityType.VERTEX));
                    for (var k = 0; k < size(q_vertices); k += 1) {
                        var q_vertex = q_vertices[k];
                        const vertex_id = transientQueriesToStrings(q_vertex);
                        edge_topo.vertices = append(edge_topo.vertices, vertex_id);

                        // 判断是否已评估该id对应的实体
                        // if (isIn(vertex_id, all_eval_ids)){continue;}
                        // all_eval_ids = append(all_eval_ids, vertex_id);

                        // 每个点仅包含：点的定义、点的id
                        var vertex_topo = {};

                        // 评估每个角点的属性
                        vertex_topo.id = vertex_id;
                        vertex_topo.param = evVertexPoint(context, {vertex: q_vertex});

                        topo.vertices = append(topo.vertices, vertex_topo);
                    }

                    topo.edges = append(topo.edges, edge_topo);

                }

                else if (isVertex)
                {
                    const vertex_id = transientQueriesToStrings(q_entity);

                    // 判断是否已评估该id对应的实体
                    // if (isIn(vertex_id, all_eval_ids)){continue;}
                    // all_eval_ids = append(all_eval_ids, vertex_id);

                    // 设置主拓扑类型
                    topo.entityType = "VERTEX";

                    // 每个点仅包含：点的定义、点的id
                    var vertex_topo = {};

                    // 评估每个角点的属性
                    vertex_topo.id = vertex_id;
                    vertex_topo.param = evVertexPoint(context, {vertex: q_entity});

                    topo.vertices = append(topo.vertices, vertex_topo);
                }

                else if(isBody)
                {
                    topo.entityType = "BODY";
                }

                else
                {
                    topo.entityType = "UNKNOWN";
                }

                res_list = append(res_list, topo);

            }
            return res_list;
            }''',
            "queries": [{"key": "id", "value": ent_id_list}]
        }

        v_list = model_url.split("/")
        did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)

        return res.json()

    def eval_bodydetails(self, did, wid, eid):
        """
        parse the B-rep representation as a dict
        """
        res = self._api.request('get', '/api/partstudios/d/{}/w/{}/e/{}/bodydetails'.format(did, wid, eid)).json()
        # extract local coordinate system for each face
        for body in res['bodies']:
            all_face_ids = [face['id'] for face in body['faces']]
            face_entity = self.get_entity_by_id(did, wid, eid, all_face_ids, 'FACE')
            face_params = self.parse_face_msg(face_entity.json()['result']['message']['value'])
            for i, face in enumerate(body['faces']):
                if face_params[i]['type'] == 'Plane':
                    x_axis = face_params[i]['x']
                elif face_params[i]['type'] == '':
                    x_axis = []
                else:
                    x_axis = face_params[i]['coordSystem']['xAxis']
                    z_axis = face_params[i]['coordSystem']['zAxis']
                    face['surface'].update({'z_axis': z_axis})
                face['surface'].update({'x_axis': x_axis})
        return res

    def eval_curve_midpoint(self, did, wid, eid, geo_id):
        """
        get the midpoint of a curve specified by its entity ID
        """
        body = {
            "script":
                "function(context is Context, queries) { " +
                "   var q_arr = evaluateQuery(context, queries.id);"
                "   var midpoint = evEdgeTangentLine(context, {\"edge\": q_arr[0], \"parameter\": 0.5 }).origin;"
                "   return midpoint;"
                "}",
            "queries": [{"key": "id", "value": [geo_id]}]
        }
        # res = c.get_entity_by_id(did, wid, eid, 'JGV', 'EDGE')
        response = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        point_info = response.json()['result']['message']['value']
        midpoint = [x['message']['value'] for x in point_info]
        return midpoint

    def expr2meter(self, did, wid, eid, expr):
        """
        convert value expresson to meter unit
        """
        body = {
            "script":
                "function(context is Context, queries) { "
                "   return lookupTableEvaluate(\"%s\") * meter;" % (expr) +
                "}",
            "queries": []
        }

        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body).json()
        return res['result']['message']['value']

