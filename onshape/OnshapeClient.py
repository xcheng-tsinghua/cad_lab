import json
import string
import random
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

            except Exception:
                self.save_creds(creds)
                exit(Fore.BLUE + f'creds is already saved, please restart the app.' + Style.RESET_ALL)

        print(Fore.GREEN + f'onshape instance created: url = {self._url}, access key = {self._access_key}' + Style.RESET_ALL)

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
            print(Fore.RED + 'request failed, details: ' + res.text + Style.RESET_ALL)

        else:
            print(Fore.WHITE + f'request succeeded, status code: {res.status_code}' + Style.RESET_ALL)

        return res


class OnshapeClient(object):
    """
    对 onshape 服务器进行一些请求操作
    """
    def __init__(self, creds='./config/onshape_credit.json'):
        """
        构造函数获取 api 秘钥，用于连接 onshape 服务器
        """
        self._api = OnshapeAPI(creds=creds)

    def request_topo_roll_back_to(self, model_url, feat_id_list, roll_back_index, is_load, json_path):
        """
        获取回滚到某个建模步骤前的模型拓扑
        回滚索引从0开始，索引为 0 表示一个特征都没有
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
                        topo.featureId = q_arr[l];
                        topo.regions = [];
                        topo.bodies = [];
                        topo.faces = [];
                        topo.edges = [];
                        topo.vertices = [];
                        
                        /* ---------- 0. Regions (regions only) ---------- */  
                        // 区域列表，每个区域仅包含：区域的定义、区域id、该区域下的边id
                        // 实测 Face 中包含 Region
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
                           topo.regions = append(topo.regions, region_topo);
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
                            
                            // 获取近似的 BSpline Surface，便于重构曲面
                            face_topo.approximateBSplineSurface = evApproximateBSplineSurface(context, {face: face_arr[i], tolerance: 1e-8});
                            
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
                            
                            // 获取近似的 BSpline Curve，便于重构曲线
                            edge_topo.approximateBSplineCurve = evApproximateBSplineCurve(context, {face: face_arr[i], tolerance: 1e-8});

                            // 获取Edge的中点，用于重构边
                            var midPoint = evEdgeTangentLine(context, {edge: edge_arr[j], parameter: 0.5}).origin;
                            edge_topo.midPoint = midPoint;

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

    def request_render_roll_back_to(self, model_url, roll_back_index: int):
        """
        将建模历史回滚到索引为roll_back_index的特征之前
        回滚索引从0开始，索引为 0 表示一个特征都没有
        """

        body = {"rollbackIndex": roll_back_index}

        v_list = model_url.split("/")
        did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

        res = self._api.request('post',
                                f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/features/rollback',
                                body=body)
        entity_topo = res.json()
        print(entity_topo)

        return entity_topo

    def request_features(self, model_url, is_load, json_path):
        """
        获取特征列表
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

