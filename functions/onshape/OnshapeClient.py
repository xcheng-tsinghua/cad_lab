import json
import string
import random
import mimetypes
import os
from colorama import Fore, Back, Style
import base64
import hashlib
from datetime import datetime, timezone
import urllib
import hmac
import requests


class OnshapeAPI(object):
    def __init__(self, creds):
        """
        Provides access to the Onshape REST API.
        Instantiates an instance of the Onshape class. Reads credentials from a JSON file
        of this format:

            {
              "onshape_url": "https://cad.onshape.com",
              "access_key": "your_access_key",
              "secret_key": "your_secret_key"
            }

        Args:
            - creds: Credentials location
        """
        assert os.path.isfile(creds), IOError(f'{creds} is not a file')

        with open(creds) as f:
            try:
                stack = json.load(f)
                self._url = stack['onshape_url']
                self._access_key = stack['access_key'].encode("utf-8")
                self._secret_key = stack['secret_key'].encode("utf-8")

            except Exception as e:
                self.save_creds(creds)
                exit(Fore.BLUE + Back.CYAN + f'Please restart the app. Exception: {e}' + Style.RESET_ALL)

        print(Fore.GREEN + Back.CYAN + f'onshape instance created: url = {self._url}, access key = {self._access_key}' + Style.RESET_ALL)

    @staticmethod
    def save_creds(json_path):
        """
        Ask user to input Onshape credentials and save them as a JSON file.

        Parameters
        ----------
        json_path : str
            Full path to the output json file, e.g. "config/onshape_auth.json"
        """

        onshape_url = input("Enter Onshape URL (e.g. https://cad.onshape.com): ").strip()
        access_key = input("Enter Onshape access_key: ").strip()
        secret_key = input("Enter Onshape secret_key: ").strip()

        config = {
            "onshape_url": onshape_url,
            "access_key": access_key,
            "secret_key": secret_key
        }

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

        print(f'Onshape credits saved to: {json_path}')

    @staticmethod
    def _make_nonce():
        """
        Generate a unique ID for the request, 25 chars in length

        Returns:
            - str: Cryptographic nonce
        """
        chars = string.digits + string.ascii_letters
        nonce = ''.join(random.choice(chars) for _ in range(25))

        return nonce

    def _make_auth(self, method, date, nonce, path, query={}, ctype='application/json'):
        """
        Create the request signature to authenticate

        Args:
            - method (str): HTTP method
            - date (str): HTTP date header string
            - nonce (str): Cryptographic nonce
            - path (str): URL pathname
            - query (dict, default={}): URL query string in key-value pairs
            - ctype (str, default='application/json'): HTTP Content-Type
        """
        query = urllib.parse.urlencode(query)
        hmac_str = (method + '\n' + nonce + '\n' + date + '\n' + ctype + '\n' + path + '\n' + query + '\n').lower()

        signature = base64.b64encode(hmac.new(self._secret_key, hmac_str.encode('utf-8'), digestmod=hashlib.sha256).digest())
        auth = 'On ' + self._access_key.decode('utf-8') + ':HmacSHA256:' + signature.decode('utf-8')

        return auth

    def _make_headers(self, method, path, query={}, headers={}):
        """
        Creates a headers object to sign the request

        Args:
            - method (str): HTTP method
            - path (str): Request path, e.g. /api/documents. No query string
            - query (dict, default={}): Query string in key-value format
            - headers (dict, default={}): Other headers to pass in

        Returns:
            - dict: Dictionary containing all headers
        """
        date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
        nonce = self._make_nonce()
        ctype = headers.get('Content-Type') if headers.get('Content-Type') else 'application/json'

        auth = self._make_auth(method, date, nonce, path, query=query, ctype=ctype)

        req_headers = {
            'Content-Type': 'application/json',
            'Date': date,
            'On-Nonce': nonce,
            'Authorization': auth,
            'User-Agent': 'Onshape Python Sample App',
            'Accept': 'application/json'
        }

        # add in user-defined headers
        for h in headers:
            req_headers[h] = headers[h]

        return req_headers

    def request(self, method, path, query={}, headers={}, body={}, base_url=None):
        """
        Issues a request to Onshape

        Args:
            - method (str): HTTP method
            - path (str): Path  e.g. /api/documents/:id
            - query (dict, default={}): Query params in key-value pairs
            - headers (dict, default={}): Key-value pairs of headers
            - body (dict, default={}): Body for POST request
            - base_url (str, default=None): Host, including scheme and port (if different from creds file)

        Returns:
            - requests.Response: Object containing the response from Onshape
        """
        req_headers = self._make_headers(method, path, query, headers)
        if base_url is None:
            base_url = self._url
        url = base_url + path + '?' + urllib.parse.urlencode(query)

        # only parse as json string if we have to
        body = json.dumps(body) if isinstance(body, dict) else body

        res = requests.request(method, url, headers=req_headers, data=body, allow_redirects=False, stream=True)

        if res.status_code == 307:
            location = urllib.parse.urlparse(res.headers["Location"])
            querystring = urllib.parse.parse_qs(location.query)

            print('request redirected to: ' + location.geturl())

            new_query = {}
            new_base_url = location.scheme + '://' + location.netloc

            for key in querystring:
                new_query[key] = querystring[key][0]  # won't work for repeated query params

            return self.request(method, location.path, query=new_query, headers=headers, base_url=new_base_url)

        elif not 200 <= res.status_code <= 206:
            print(Fore.RED + Back.CYAN + 'request failed, details: ' + res.text + Style.RESET_ALL)

        else:
            print(Fore.WHITE + Back.CYAN + f'request succeeded, status code: {res.status_code}' + Style.RESET_ALL)

        return res


class OnshapeClient(object):
    """
    inherited from OnShape public apikey python client,
    with additional method for parsing cad.
    """
    def __init__(self, creds='./config/onshape_credit.json'):
        """
        Instantiates a new Onshape client.

        Args:
            - stack (str, default='https://cad.onshape.com'): Base URL
            - logging (bool, default=True): Turn logging on or off
        """
        self._api = OnshapeAPI(creds=creds)

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
        return self._api.request('get', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/tessellatedfaces')

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
        res = self._api.request('post', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript', body=body)
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
        res = self._api.request('post', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript', body=body)
        return res

    def request_multi_feat_topology(self, model_url, feat_id_list, is_load, json_path):
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
                '''
                function(context is Context, queries) { 
                    var res_list = [];
                    var q_arr = [''' + ",".join([f"\"{fid}\"" for fid in feat_id_list]) + "];" +
                '''
                    for (var l = 0; l < size(q_arr); l+= 1){
                        var topo = {};
                        topo.faces = [];
                        topo.edges = [];
                        topo.vertices = [];
                    
                        /* ---------- 1. Face (ALL faces generated) ---------- */
                        var q_face = qCreatedBy(makeId(q_arr[l]), EntityType.FACE);
                        var face_arr = evaluateQuery(context, q_face);
                        for (var i = 0; i < size(face_arr); i += 1) {
                            var face_topo = {};
                            face_topo.id = transientQueriesToStrings(face_arr[i]);
                            face_topo.edges = [];
                            face_topo.param = evSurfaceDefinition(context, {face: face_arr[i]});
                            var q_edge = qAdjacent(face_arr[i], AdjacencyType.EDGE, EntityType.EDGE);
                            var edge_arr = evaluateQuery(context, q_edge);
                            for (var j = 0; j < size(edge_arr); j += 1) {
                                const edge_id = transientQueriesToStrings(edge_arr[j]);
                                face_topo.edges = append(face_topo.edges, edge_id);
                            }
                            topo.faces = append(topo.faces, face_topo);
                        }
                                                                                                        
                        /* ---------- 2. Edges (ALL sketch edges, open or closed) ---------- */
                        var q_edge = qCreatedBy(makeId(q_arr[l]), EntityType.EDGE);
                        var edge_arr = evaluateQuery(context, q_edge);
                        for (var j = 0; j < size(edge_arr); j += 1) {
                            var edge_topo = {};
                            const edge_id = transientQueriesToStrings(edge_arr[j]);
                            edge_topo.id = edge_id;
                            edge_topo.vertices = [];
                            edge_topo.param = evCurveDefinition(context, {edge: edge_arr[j]});
                            
                            // obtain the midpoint of the edge
                            var midpoint = evEdgeTangentLine(context, {edge: edge_arr[j], parameter: 0.5 }).origin;
                            edge_topo.midpoint = midpoint;
                            
                            var q_vertex = qAdjacent(edge_arr[j], AdjacencyType.VERTEX, EntityType.VERTEX);
                            var vertex_arr = evaluateQuery(context, q_vertex);
                            for (var k = 0; k < size(vertex_arr); k += 1) {
                                const vertex_id = transientQueriesToStrings(vertex_arr[k]);
                                edge_topo.vertices = append(edge_topo.vertices, vertex_id);
                            }
                            topo.edges = append(topo.edges, edge_topo);                                                           
                        }
                                                                                                        
                       /* ---------- 3. Vertices (ALL sketch vertices) ---------- */
                        var q_vertex = qCreatedBy(makeId(q_arr[l]), EntityType.VERTEX);
                        var vertex_arr = evaluateQuery(context, q_vertex);
                        for (var k = 0; k < size(vertex_arr); k += 1) {
                            var vertex_topo = {};
                            const vertex_id = transientQueriesToStrings(vertex_arr[k]);
                            vertex_topo.id = vertex_id;
                            vertex_topo.param = evVertexPoint(context, {vertex: vertex_arr[k]});
                            topo.vertices = append(topo.vertices, vertex_topo);
                        }
                        res_list = append(res_list, topo);
                    }
                    return res_list;
                }
                ''',
            "queries": []
        }

        if is_load:
            print(f'从文件加载原始拓扑列表: {json_path}')
            with open(json_path, 'r') as f:
                entity_topo = json.load(f)

        else:
            print('从 onshape 请求原始拓扑列表')

            v_list = model_url.split("/")
            did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

            res = self._api.request('post',
                                    f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript',
                                    body=body)
            entity_topo = res.json()

            print('保存原始拓扑列表')
            with open(json_path, 'w') as f:
                json.dump(entity_topo, f, ensure_ascii=False, indent=4)

        return entity_topo

    def request_topo_roll_back_to(self, model_url, feat_id_list, roll_back_index, is_load, json_path):
        """
        获取回滚到某个建模步骤前的模型拓扑
        :param model_url:
        :param feat_id_list:
        :param roll_back_index:
        :param is_load:
        :param json_path:
        :return:
        """
        body = {
            "script":
                '''
                function(context is Context, queries) { 
                    var res_list = [];
                    var q_arr = [''' + ",".join([f"\"{fid}\"" for fid in feat_id_list]) + "];" +
                '''
                    for (var l = 0; l < size(q_arr); l+= 1){
                        var topo = {};
                        topo.bodies = [];
                        topo.faces = [];
                        topo.edges = [];
                        topo.vertices = [];
                        
                        /* ---------- 0. Regions (regions only) ---------- */  // 区域列表，每个区域仅包含：区域的定义、区域id、该区域下的边id
                        var q_region = qSketchRegion(makeId(q_arr[l]));
                        var region_arr = evaluateQuery(context, q_region);
                        for (var i = 0; i < size(region_arr); i += 1) {
                           var region_topo = {};
                           region_topo.id = transientQueriesToStrings(region_arr[i]);  // 区域id
                           region_topo.edges = [];  // 该区域下的边id
                           region_topo.param = evSurfaceDefinition(context, {face: region_arr[i]});  // 区域的定义
                           var q_edge = qAdjacent(region_arr[i], AdjacencyType.EDGE, EntityType.EDGE);
                           var edge_arr = evaluateQuery(context, q_edge);
                           for (var j = 0; j < size(edge_arr); j += 1) {
                               const edge_id = transientQueriesToStrings(edge_arr[j]);
                               region_topo.edges = append(region_topo.edges, edge_id);
                           }
                           topo.faces = append(topo.faces, region_topo);
                        }
                        
                        /* ---------- 1. Body (ALL bodies generated) ---------- */
                        var q_body = qCreatedBy(makeId(q_arr[l]), EntityType.BODY);
                        var body_arr = evaluateQuery(context, q_body);
                        for (var i = 0; i < size(body_arr); i += 1) {
                            var body_topo = {};
                            body_topo.id = transientQueriesToStrings(body_arr[i]);
                            body_topo.faces = [];
                            var q_face = qOwnedByBody(body_arr[i], EntityType.FACE);
                            var face_arr = evaluateQuery(context, q_face);
                            for (var j = 0; j < size(face_arr); j += 1) {
                                const face_id = transientQueriesToStrings(face_arr[j]);
                                body_topo.faces = append(body_topo.faces, face_id);
                            }
                            topo.bodies = append(topo.bodies, body_topo);
                        }

                        /* ---------- 1. Face (ALL faces generated) ---------- */
                        var q_face = qCreatedBy(makeId(q_arr[l]), EntityType.FACE);
                        var face_arr = evaluateQuery(context, q_face);
                        for (var i = 0; i < size(face_arr); i += 1) {
                            var face_topo = {};
                            face_topo.id = transientQueriesToStrings(face_arr[i]);
                            face_topo.edges = [];
                            face_topo.param = evSurfaceDefinition(context, {face: face_arr[i]});
                            var q_edge = qAdjacent(face_arr[i], AdjacencyType.EDGE, EntityType.EDGE);
                            var edge_arr = evaluateQuery(context, q_edge);
                            for (var j = 0; j < size(edge_arr); j += 1) {
                                const edge_id = transientQueriesToStrings(edge_arr[j]);
                                face_topo.edges = append(face_topo.edges, edge_id);
                            }
                            topo.faces = append(topo.faces, face_topo);
                        }

                        /* ---------- 2. Edges (ALL sketch edges, open or closed) ---------- */
                        var q_edge = qCreatedBy(makeId(q_arr[l]), EntityType.EDGE);
                        var edge_arr = evaluateQuery(context, q_edge);
                        for (var j = 0; j < size(edge_arr); j += 1) {
                            var edge_topo = {};
                            const edge_id = transientQueriesToStrings(edge_arr[j]);
                            edge_topo.id = edge_id;
                            edge_topo.vertices = [];
                            edge_topo.param = evCurveDefinition(context, {edge: edge_arr[j]});

                            // obtain the midpoint of the edge
                            var midpoint = evEdgeTangentLine(context, {edge: edge_arr[j], parameter: 0.5 }).origin;
                            edge_topo.midpoint = midpoint;

                            var q_vertex = qAdjacent(edge_arr[j], AdjacencyType.VERTEX, EntityType.VERTEX);
                            var vertex_arr = evaluateQuery(context, q_vertex);
                            for (var k = 0; k < size(vertex_arr); k += 1) {
                                const vertex_id = transientQueriesToStrings(vertex_arr[k]);
                                edge_topo.vertices = append(edge_topo.vertices, vertex_id);
                            }
                            topo.edges = append(topo.edges, edge_topo);                                                           
                        }

                       /* ---------- 3. Vertices (ALL sketch vertices) ---------- */
                        var q_vertex = qCreatedBy(makeId(q_arr[l]), EntityType.VERTEX);
                        var vertex_arr = evaluateQuery(context, q_vertex);
                        for (var k = 0; k < size(vertex_arr); k += 1) {
                            var vertex_topo = {};
                            const vertex_id = transientQueriesToStrings(vertex_arr[k]);
                            vertex_topo.id = vertex_id;
                            vertex_topo.param = evVertexPoint(context, {vertex: vertex_arr[k]});
                            topo.vertices = append(topo.vertices, vertex_topo);
                        }
                        res_list = append(res_list, topo);
                    }
                    return res_list;
                }
                ''',
            "queries": []
        }

        if is_load:
            print(Fore.GREEN + f'从文件加载原始拓扑列表: {json_path}' + Style.RESET_ALL)
            with open(json_path, 'r') as f:
                entity_topo = json.load(f)

        else:
            print(Fore.CYAN + '从 onshape 请求原始拓扑列表' + Style.RESET_ALL)

            v_list = model_url.split("/")
            did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

            res = self._api.request('post',
                                    f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript',
                                    {'rollbackBarIndex': roll_back_index}, body=body)
            entity_topo = res.json()

            print(Fore.CYAN + '保存原始拓扑列表' + Style.RESET_ALL)
            with open(json_path, 'w') as f:
                json.dump(entity_topo, f, ensure_ascii=False, indent=4)

        return entity_topo

    def request_set_roll_back_to(self, model_url, roll_back_index: int):
        """
        将建模历史回滚到索引为roll_back_index的特征之前，索引从0开始

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - feat_id (str): Feature ID of a sketch or operation

        Returns:
            - dict: a hierarchical parametric representation
        """

        body = {"rollbackIndex": roll_back_index}

        v_list = model_url.split("/")
        did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

        res = self._api.request('post', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/features/rollback',
                                body=body)
        entity_topo = res.json()
        print(entity_topo)

        return entity_topo

    def request_multi_entity_topology(self, model_url, ent_id_list, is_load, json_path):
        """
        通过草图特征或者拉伸等特征的 id 解析拓扑结构，包含草图区域，边、角点等

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - feat_id (str): Feature ID of a sketch or operation

        Returns:
            - dict: a hierarchical parametric representation


            # var q_entity_list = [''' + ",".join([f"\"{fid}\"" for fid in ent_id_list]) + '''];


        """
        body = {
            'script': '''function(context is Context, queries) {
            var all_eval_ids = []; // 已评估的id，不会重复评估
            var res_list = [];  // 最终返回的结果列表
            var q_entity_list = evaluateQuery(context, queries.id);
            
            var entity_num = {};
            entity_num.n_entity = size(q_entity_list);
            res_list = append(res_list, entity_num);
            
            for (var i = 0; i < size(q_entity_list); i+= 1){ // entityId 食欲个包含了诸如 JDC、JGI的id列表
                var q_entity = q_entity_list[i];

                const entity_id = transientQueriesToStrings(q_entity);

                var topo = {};  // 设置当前id对应的属性
                topo.index = i;
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
                    if (isIn(face_id, all_eval_ids)){continue;}
                    all_eval_ids = append(all_eval_ids, face_id);

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
                        if (isIn(edge_id, all_eval_ids)){continue;}
                        all_eval_ids = append(all_eval_ids, edge_id);

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
                            if (isIn(vertex_id, all_eval_ids)){continue;}
                            all_eval_ids = append(all_eval_ids, vertex_id);

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
                    if (isIn(edge_id, all_eval_ids)){continue;}
                    all_eval_ids = append(all_eval_ids, edge_id);

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
                        if (isIn(vertex_id, all_eval_ids)){continue;}
                        all_eval_ids = append(all_eval_ids, vertex_id);

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
                    if (isIn(vertex_id, all_eval_ids)){continue;}
                    all_eval_ids = append(all_eval_ids, vertex_id);

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

        if is_load:
            print(f'从文件加载原始拓扑列表: {json_path}')
            with open(json_path, 'r') as f:
                entity_topo = json.load(f)

        else:
            print('从 onshape 请求原始拓扑列表')

            v_list = model_url.split("/")
            did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
            res = self._api.request('post', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript', body=body)

            entity_topo = res.json()

            print('保存原始拓扑列表')
            with open(json_path, 'w') as f:
                json.dump(entity_topo, f, ensure_ascii=False, indent=4)

        return entity_topo

    def request_multi_entity_topology_v2(self, model_url, ent_id_list, is_load, json_path):
        """
        通过草图特征或者拉伸等特征的 id 解析拓扑结构，包含草图区域，边、角点等

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - feat_id (str): Feature ID of a sketch or operation

        Returns:
            - dict: a hierarchical parametric representation


            # var q_entity_list = [''' + ",".join([f"\"{fid}\"" for fid in ent_id_list]) + '''];


        """
        body = {
            'script': '''function(context is Context, queries) {
            var all_eval_ids = [];
            var res_list = [];
            var q_entity_list = [];
            ''' +

            "\n".join(f"""var q_entity{i} = evaluateQuery(context, queries.id{i}); 
            if (size(q_entity{i}) == 1) {{
            q_entity_list = append(q_entity_list, q_entity{i}[0]);}}""" for i in range(len(ent_id_list))) +
            '''
            
            var entity_num = {};
            entity_num.n_entity = size(q_entity_list);
            res_list = append(res_list, entity_num);

            for (var i = 0; i < size(q_entity_list); i+= 1){
                var q_entity = q_entity_list[i];
                const entity_id = transientQueriesToStrings(q_entity);

                var topo = {};
                topo.index = i;
                topo.id = entity_id;
                topo.faces = [];
                topo.edges = [];
                topo.vertices = [];

                var isBody = size(evaluateQuery(context, qEntityFilter(q_entity, EntityType.BODY))) == 1;
                var isFace = size(evaluateQuery(context, qEntityFilter(q_entity, EntityType.FACE))) == 1;
                var isEdge = size(evaluateQuery(context, qEntityFilter(q_entity, EntityType.EDGE))) == 1;
                var isVertex = size(evaluateQuery(context, qEntityFilter(q_entity, EntityType.VERTEX))) == 1;

                if (isFace)
                {
                    const face_id = transientQueriesToStrings(q_entity);

                    if (isIn(face_id, all_eval_ids)){continue;}
                    all_eval_ids = append(all_eval_ids, face_id);

                    topo.entityType = "FACE";

                    var face_topo = {};
                    face_topo.id = face_id;
                    face_topo.param = evSurfaceDefinition(context, {face: q_entity});
                    face_topo.edges = [];

                    var q_edges = evaluateQuery(context, qAdjacent(q_entity, AdjacencyType.EDGE, EntityType.EDGE));
                    for (var j = 0; j < size(q_edges); j += 1) {
                        var q_edge = q_edges[j];
                        const edge_id = transientQueriesToStrings(q_edge);
                        face_topo.edges = append(face_topo.edges, edge_id);

                        if (isIn(edge_id, all_eval_ids)){continue;}
                        all_eval_ids = append(all_eval_ids, edge_id);

                        var edge_topo = {};
                        edge_topo.id = edge_id;
                        edge_topo.param = evCurveDefinition(context, {edge: q_edge});
                        edge_topo.vertices = [];

                        var q_vertices = evaluateQuery(context, qAdjacent(q_edge, AdjacencyType.VERTEX, EntityType.VERTEX));
                        for (var k = 0; k < size(q_vertices); k += 1) {
                            var q_vertex = q_vertices[k];
                            const vertex_id = transientQueriesToStrings(q_vertex);
                            edge_topo.vertices = append(edge_topo.vertices, vertex_id);

                            if (isIn(vertex_id, all_eval_ids)){continue;}
                            all_eval_ids = append(all_eval_ids, vertex_id);

                            var vertex_topo = {};

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
                    const edge_id = transientQueriesToStrings(q_entity);

                    if (isIn(edge_id, all_eval_ids)){continue;}
                    all_eval_ids = append(all_eval_ids, edge_id);

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

                        if (isIn(vertex_id, all_eval_ids)){continue;}
                        all_eval_ids = append(all_eval_ids, vertex_id);

                        var vertex_topo = {};

                        vertex_topo.id = vertex_id;
                        vertex_topo.param = evVertexPoint(context, {vertex: q_vertex});

                        topo.vertices = append(topo.vertices, vertex_topo);
                    }

                    topo.edges = append(topo.edges, edge_topo);

                }

                else if (isVertex)
                {
                    const vertex_id = transientQueriesToStrings(q_entity);

                    if (isIn(vertex_id, all_eval_ids)){continue;}
                    all_eval_ids = append(all_eval_ids, vertex_id);

                    topo.entityType = "VERTEX";

                    var vertex_topo = {};

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
            "queries": [{"key": f"id{i}", "value": [ent_id]} for i, ent_id in enumerate(ent_id_list)]
        }

        if is_load:
            print(f'从文件加载原始拓扑列表: {json_path}')
            with open(json_path, 'r') as f:
                entity_topo = json.load(f)

        else:
            print('从 onshape 请求原始拓扑列表')

            v_list = model_url.split("/")
            did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
            res = self._api.request('post', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript',
                                    body=body)

            entity_topo = res.json()

            print('保存原始拓扑列表')
            with open(json_path, 'w') as f:
                json.dump(entity_topo, f, ensure_ascii=False, indent=4)

        return entity_topo

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
        response = self._api.request('post', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript', body=body)
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

        res = self._api.request('post', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript', body=body).json()
        return res['result']['message']['value']

    def request_features(self, model_url, is_load, json_path):
        """
        Gets the feature list for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        """
        if is_load:
            print(Fore.GREEN + f'从文件加载原始特征列表: {json_path}' + Style.RESET_ALL)
            with open(json_path, 'r') as f:
                ofs = json.load(f)

        else:
            print(Fore.CYAN + '从 onshape 请求原始特征列表' + Style.RESET_ALL)

            v_list = model_url.split("/")
            did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

            res = self._api.request('get', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/features')
            ofs = res.json()

            print(Fore.CYAN + '保存原始特征列表' + Style.RESET_ALL)
            with open(json_path, 'w') as f:
                json.dump(ofs, f, ensure_ascii=False, indent=4)

        return ofs

    def request_body_details(self, model_url, roll_back_index, is_load, json_path):
        """
        从 onshape 服务器请求现有实体
        :param model_url:
        :param roll_back_index:
        :param is_load:
        :param json_path:
        :return:
        """
        if is_load:
            print(Fore.GREEN + f'从文件加载原始模型特征列表: {json_path}' + Style.RESET_ALL)
            with open(json_path, 'r') as f:
                ofs = json.load(f)

        else:
            print(Fore.CYAN + '从 onshape 请求原始特征列表' + Style.RESET_ALL)

            v_list = model_url.split("/")
            did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

            res = self._api.request('get', f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/bodydetails', {'rollbackBarIndex': roll_back_index})
            ofs = res.json()

            print(Fore.CYAN + '保存原始模型特征列表' + Style.RESET_ALL)
            with open(json_path, 'w') as f:
                json.dump(ofs, f, ensure_ascii=False, indent=4)

        return ofs




