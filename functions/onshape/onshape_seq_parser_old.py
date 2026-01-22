"""
登录onshape的配置文件：
./config/onshape_credit.json

文件内容示例：
{
  "onshape_url": "https://cad.onshape.com",
  "access_key": "your_access_key",
  "secret_key": "your_secret_key"
}
"""

import os
import json
import numpy as np
import copy
from collections import OrderedDict
import math
import string
import random
import mimetypes
import base64
import hashlib
from datetime import datetime, timezone
import urllib
import hmac
import requests
import sys
from functions.onshape import macro


EXTENT_TYPE_MAP = {'BLIND': 'OneSideFeatureExtentType',
                   'SYMMETRIC': 'SymmetricFeatureExtentType'}

OPERATION_MAP = {'NEW': 'NewBodyFeatureOperation',
                 'ADD': 'JoinFeatureOperation',
                 'REMOVE': 'CutFeatureOperation',
                 'INTERSECT': 'IntersectFeatureOperation'}


def xyz_list2dict(xyz):
    return OrderedDict({'x': xyz[0], 'y': xyz[1], 'z': xyz[2]})


def angle_from_vector_to_x(vec):
    """
    对于输入的2维单位向量 vec，确定其与 (0, 1) 之间的夹角
    # 2 | 1
    # -------
    # 3 | 4

    :param vec:
    :return:
    """
    if vec[0] >= 0:
        if vec[1] >= 0:
            # 第 1 象限
            angle = math.asin(vec[1])
        else:
            # 第 4 象限
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # 第 2 象限
            angle = math.pi - math.asin(vec[1])
        else:
            # 第 3 象限
            angle = math.pi + math.asin(-vec[1])
    return angle


class Onshape(object):
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
        if not os.path.isfile(creds):
            raise IOError(f'{creds} is not a file')

        with open(creds) as f:
            try:
                stack = json.load(f)
                self._url = stack['onshape_url']
                self._access_key = stack['access_key'].encode("utf-8")
                self._secret_key = stack['secret_key'].encode("utf-8")

            except Exception as e:
                self.save_creds(creds)
                exit(f'Please restart the app. Exception: {e}')

        print(f'onshape instance created: url = {self._url}, access key = {self._access_key}')

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

        # print('nonce created: %s' % nonce)

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

        hmac_str = (method + '\n' + nonce + '\n' + date + '\n' + ctype + '\n' + path +
                    '\n' + query + '\n').lower()

        signature = base64.b64encode(hmac.new(self._secret_key, hmac_str.encode('utf-8'), digestmod=hashlib.sha256).digest())
        auth = 'On ' + self._access_key.decode('utf-8') + ':HmacSHA256:' + signature.decode('utf-8')

        # print({'query': query,
        #        'hmac_str': hmac_str,
        #        'signature': signature,
        #        'auth': auth})

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
        # time.sleep(1.0)  # 防止因请求过快导致请求失败

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
            print('request failed, details: ' + res.text)
            # print('request failed')

        else:
            # print('request succeeded, details: ' + res.text)
            print(f'request succeeded, status code: {res.status_code}')

        return res


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
        self._api = Onshape(creds=creds)

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

    def get_features(self, did, wid, eid):
        """
        Gets the feature list for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        """
        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features')

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


class MyClient(Client):
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
            "queries": [{ "key" : "id", "value" : geo_id }]
        }
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'get_ent_by_id_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(res.json(), f, ensure_ascii=False, indent=4)

        return res

    def eval_sketch_topology_by_adjacency(self, did, wid, eid, feat_id):
        """
        parse the hierarchical parametric geometry&topology (face -> edges -> vertex)
        from a specified sketch feature ID.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - feat_id (str): Feature ID of a sketch

        Returns:
            - dict: a hierarchical parametric representation
        """
        body = {
            "script":
                "function(context is Context, queries) { "
                "   var topo = {};"
                "   topo.faces = [];"
                "   topo.edges = [];"
                "   topo.vertices = [];"
                "   var all_edge_ids = [];"
                "   var all_vertex_ids = [];"
                "                           "
                "   var q_face = qSketchRegion(makeId(\"%s\"));" % feat_id +
                # "   var q_face = qCreatedBy(makeId(\"%s\"), EntityType.FACE);" % feat_id +
                "   var face_arr = evaluateQuery(context, q_face);"
                "   for (var i = 0; i < size(face_arr); i += 1) {"
                "       var face_topo = {};"
                "       const face_id = transientQueriesToStrings(face_arr[i]);"
                "       face_topo.id = face_id;"
                "       face_topo.edges = [];"
                "       face_topo.param = evSurfaceDefinition(context, {face: face_arr[i]});"
                "                            "
                # "       var q_edge = qLoopEdges(q_face);"
                "       var q_edge = qAdjacent(face_arr[i], AdjacencyType.EDGE, EntityType.EDGE);"
                "       var edge_arr = evaluateQuery(context, q_edge);"
                "       for (var j = 0; j < size(edge_arr); j += 1) {"
                "           var edge_topo = {};"
                "           const edge_id = transientQueriesToStrings(edge_arr[j]);"
                "           edge_topo.id = edge_id;"
                "           edge_topo.vertices = [];"
                "           edge_topo.param = evCurveDefinition(context, {edge: edge_arr[j]});" # 
                "           face_topo.edges = append(face_topo.edges, edge_id);"
                "                                  "
                "           var q_vertex = qAdjacent(edge_arr[j], AdjacencyType.VERTEX, EntityType.VERTEX);"
                "           var vertex_arr = evaluateQuery(context, q_vertex);"
                "           for (var k = 0; k < size(vertex_arr); k += 1) {"
                "               var vertex_topo = {};"
                "               const vertex_id = transientQueriesToStrings(vertex_arr[k]);"
                "               vertex_topo.id = vertex_id;"
                "               vertex_topo.param = evVertexPoint(context, {vertex: vertex_arr[k]});"
                "               edge_topo.vertices = append(edge_topo.vertices, vertex_id);"
                "               if (isIn(vertex_id, all_vertex_ids)){continue;}"
                "               all_vertex_ids = append(all_vertex_ids, vertex_id);"
                "               topo.vertices = append(topo.vertices, vertex_topo);"
                "           }"
                "           if (isIn(edge_id, all_edge_ids)){continue;}"
                "           all_edge_ids = append(all_edge_ids, edge_id);"
                "           topo.edges = append(topo.edges, edge_topo);"
                "       }"
                "       topo.faces = append(topo.faces, face_topo);"
                "   }"
                "   return topo;"
                "}",
            "queries": []
        }
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        res_msg = res.json()['result']['message']['value']

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'sketch_topology_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(res.json(), f, ensure_ascii=False, indent=4)

        topo = {}
        for item in res_msg:
            k_str = item['message']['key']['message']['value']  # faces, edges
            v_item = item['message']['value']['message']['value']
            outer_list = []

            for item_x in v_item:
                v_item_x = item_x['message']['value']
                geo_dict = {}

                for item_y in v_item_x:
                    k = item_y['message']['key']['message']['value']  # id, edges/vertices
                    v_msg = item_y['message']['value']

                    if k == 'param':
                        if k_str == 'faces':
                            v = MyClient.parse_face_msg(v_msg)[0]

                        elif k_str == 'edges':
                            v = MyClient.parse_edge_msg(v_msg)[0]

                        elif k_str == 'vertices':
                            v = MyClient.parse_vertex_msg(v_msg)[0]

                        else:
                            raise ValueError

                    elif isinstance(v_msg['message']['value'], list):
                        v = [a['message']['value'] for a in v_msg['message']['value']]

                    else:
                        v = v_msg['message']['value']

                    geo_dict.update({k: v})
                outer_list.append(geo_dict)
            topo.update({k_str: outer_list})
        return topo

    @staticmethod
    def parse_vertex_msg(response):
        """parse vertex parameters from OnShape response data"""
        # data = response.json()['result']['message']['value']
        data = [response] if not isinstance(response, list) else response
        vertices = []

        for item in data:
            xyz_msg = item['message']['value']
            xyz_type = item['message']['typeTag']
            p = []

            for msg in xyz_msg:
                p.append(round(msg['message']['value'], 8))

            unit = xyz_msg[0]['message']['unitToPower'][0]
            unit_exp = (unit['key'], unit['value'])
            vertices.append({xyz_type: tuple(p), 'unit': unit_exp})

        return vertices

    @staticmethod
    def parse_coord_msg(response):
        """parse coordSystem parameters from OnShape response data"""
        coord_param = {}
        for item in response:
            k_msg = item['message']['key']
            k = k_msg['message']['value']

            v_msg = item['message']['value']
            v = [round(x['message']['value'], 8) for x in v_msg['message']['value']]

            coord_param[k] = v
        return coord_param

    @staticmethod
    def parse_edge_msg(response):
        """
        parse edge parameters from OnShape response data
        """
        # data = response.json()['result']['message']['value']
        data = [response] if not isinstance(response, list) else response

        # with open(r'E:\document\DeeplearningIdea\multi_cmd_seq_gen\edgeinfo.json', 'w') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)
        # exit('--------------')

        edges = []
        for item in data:
            edge_msg = item['message']['value']
            edge_type = item['message']['typeTag']
            edge_param = {'type': edge_type}
            for msg in edge_msg:
                k = msg['message']['key']['message']['value']
                v_item = msg['message']['value']['message']['value']
                if k == 'coordSystem':
                    v = MyClient.parse_coord_msg(v_item)
                elif isinstance(v_item, list):
                    v = [round(x['message']['value'], 8) for x in v_item]
                else:
                    if isinstance(v_item, float):
                        v = round(v_item, 8)
                    else:
                        v = v_item
                edge_param.update({k: v})
            edges.append(edge_param)
        return edges

    @staticmethod
    def parse_face_msg(response):
        """
        parse face parameters from OnShape response data
        """
        # data = response.json()['result']['message']['value']
        data = [response] if not isinstance(response, list) else response
        faces = []

        for item in data:
            face_msg = item['message']['value']
            face_type = item['message']['typeTag']
            face_param = {'type': face_type}

            for msg in face_msg:
                k = msg['message']['key']['message']['value']
                v_item = msg['message']['value']['message']['value']
                if k == 'coordSystem':
                    v = MyClient.parse_coord_msg(v_item)

                elif isinstance(v_item, list):
                    v = [round(x['message']['value'], 8) for x in v_item]

                else:
                    if isinstance(v_item, float):
                        v = round(v_item, 8)

                    else:
                        v = v_item

                face_param[k] = v
            faces.append(face_param)
        return faces

    def eval_entityID_created_by_feature(self, did, wid, eid, feat_id, entity_type):
        """
        get IDs of all geometry entity created by a given feature, with specified type

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            feat_id (str): Feature ID
            entity_type (str): 'VERTEX', 'EDGE', 'FACE', 'BODY'

        Returns:
            list: a list of entity IDs
        """
        if entity_type not in ['VERTEX', 'EDGE', 'FACE', 'BODY']:
            raise ValueError("Got entity_type: %s" % entity_type)
        body = {
            "script":
                "function(context is Context, queries) { "
                "   return transientQueriesToStrings("
                "       evaluateQuery(context, " +
                "           qCreatedBy(makeId(\"%s\"), EntityType.%s)" % (feat_id, entity_type) +
                "       )"
                "   );"
                "}",
            "queries": []
        }
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'entity_ids_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(res.json(), f, ensure_ascii=False, indent=4)

        res_msg = res.json()['result']['message']['value']
        entityIDs = [item['message']['value'].encode(encoding='UTF-8') for item in res_msg]
        return entityIDs

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

    def eval_bounding_box(self, did, wid, eid):
        """
        Get bounding box of all solid bodies for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - dict: {'maxCorner': [], 'minCorner': []}
        """
        body = {
            "script":
                "function(context is Context, queries) { " +
                "   var q_body = qBodyType(qEverything(EntityType.BODY), BodyType.SOLID);"
                "   var bbox = evBox3d(context, {'topology': q_body});"
                "   return bbox;"
                "}",
            "queries": []
        }
        response = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'bnd_box_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=4)

        bbox_values = response.json()['result']['message']['value']
        result = {}
        for item in bbox_values:
            k = item['message']['key']['message']['value']
            point_values = item['message']['value']['message']['value']

            v = [x['message']['value'] for x in point_values]
            result[k] = v

        return result

    def eval_curveLength(self, did, wid, eid, geo_id):
        """
        get the length of a curve specified by its entity ID
        """
        body = {
            "script":
                "function(context is Context, queries) { " +
                "   var res_list = [];"
                "   var q_arr = evaluateQuery(context, queries.id);"
                "   for (var i = 0; i < size(q_arr); i+= 1){"
                "       var res = evLength(context, {\"entities\": q_arr[i]});"
                "       res_list = append(res_list, res);"
                "   }"
                "   return res_list;"
                "}",
            "queries": [{"key": "id", "value": [geo_id]}]
        }
        # res = c.get_entity_by_id(did, wid, eid, 'JGV', 'EDGE')
        response = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        edge_len = response.json()['result']['message']['value'][0]['message']['value']
        return edge_len

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

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'mid_point_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=4)

        point_info = response.json()['result']['message']['value']
        midpoint = [x['message']['value'] for x in point_info]
        return midpoint

    def expr2meter(self, did, wid, eid, expr):
        """
        convert value expresson to meter unit
        """
        val, unit = expr.split(' ')

        if unit == 'in':
            val = float(val) * macro.IN_TO_METER

        elif unit == 'METER':
            val = float(val)

        else:
            raise NotImplementedError

        return val

        float_val = float([0])
        print(f'{expr} trans to {float_val}')
        return float_val

        body = {
            "script":
                "function(context is Context, queries) { "
                "   return lookupTableEvaluate(\"%s\") * meter;" % (expr) +
                "}",
            "queries": []
        }

        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body).json()
        return res['result']['message']['value']


class FeatureListParser(object):
    def __init__(self, client, did, wid, eid, orig_ofs=None):
        """
        A parser for OnShape feature list (construction sequence)
        :param client:
        :param did:
        :param wid:
        :param eid:
        :param orig_ofs:
        """
        self.client = client

        self.did = did
        self.wid = wid
        self.eid = eid

        if orig_ofs is None:
            print('get feature_list by onshape_api request')
            self.feature_list = self.client.get_features(did, wid, eid).json()
        else:
            print('load feature_list')
            self.feature_list = orig_ofs

        self.profile2sketch = {}

    @staticmethod
    def parse_feature_param(fea_item_msg_param_ofs):
        param_dict = {}
        for i, param_item in enumerate(fea_item_msg_param_ofs):
            param_msg = param_item['message']
            param_id = param_msg['parameterId']  # sketchPlane

            if 'queries' in param_msg:
                param_value = []
                for j in range(len(param_msg['queries'])):
                    param_value.extend(param_msg['queries'][j]['message']['geometryIds'])  # [JDC]

            elif 'expression' in param_msg:
                param_value = param_msg['expression']

            elif 'value' in param_msg:
                param_value = param_msg['value']

            else:
                raise NotImplementedError('param_msg:\n{}'.format(param_msg))

            param_dict[param_id] = param_value
        return param_dict

    def _parse_sketch(self, fea_item_msg_ofs):
        sket_parser = SketchParser(self.client, fea_item_msg_ofs, self.did, self.wid, self.eid)  # 包含 api 请求
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

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'parse_extrude_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)

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

        for i, feat_item in enumerate(self.feature_list['features']):
            fea_item_msg_ofs = feat_item['message']
            feat_type = fea_item_msg_ofs['featureType']
            feat_id = fea_item_msg_ofs['featureId']

            if feat_type == 'newSketch':
                feat_dict = self._parse_sketch(fea_item_msg_ofs)  # 包含 api 请求
                for pf_key in feat_dict['profiles'].keys():
                    self.profile2sketch[pf_key] = feat_id

            elif feat_type == 'extrude':
                feat_dict = self._parse_extrude(fea_item_msg_ofs)
                feat_dict['type'] = 'None'

            elif feat_type == 'revolve':  # 旋转
                feat_dict = self._parse_revolve(fea_item_msg_ofs)
                feat_dict['type'] = 'None'

            elif feat_type == 'loft':  # 放样
                feat_dict = self._parse_loft(fea_item_msg_ofs)
                feat_dict['type'] = 'None'

            elif feat_type == 'sweep':  # 扫描
                feat_dict = self._parse_sweep(fea_item_msg_ofs)
                feat_dict['type'] = 'None'

            elif feat_type == '线性阵列、圆周阵列？':
                feat_dict = {}
                feat_dict['type'] = 'None'

            else:
                print(f'unsupported feature type: {feat_type}')
                feat_dict = {}
                feat_dict['type'] = 'None'
                continue
                raise NotImplementedError("unsupported feature type: {}".format(feat_type))

            result["entities"][feat_id] = feat_dict
            result["sequence"].append({"index": i, "type": feat_dict['type'], "entity": feat_id})

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'parse_res_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return result


class SketchParser(object):
    """
    A parser for OnShape sketch feature list
    """
    def __init__(self, client, fea_item_msg_ofs, did, wid, eid):
        self.client = client
        self.feat_id = fea_item_msg_ofs['featureId']  # FlWH1mrqLpBuR3O_0
        self.feat_name = fea_item_msg_ofs['name']  # Sketch 1
        self.feat_param = FeatureListParser.parse_feature_param(fea_item_msg_ofs['parameters'])

        self.did = did
        self.wid = wid
        self.eid = eid

        geo_id = self.feat_param["sketchPlane"][0]  # JDC
        response = self.client.get_entity_by_id(did, wid, eid, [geo_id], "FACE")  # 包含 api 请求

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'face_resp_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=4)

        self.plane = self.client.parse_face_msg(response.json()['result']['message']['value'])[0]

        self.geo_topo = self.client.eval_sketch_topology_by_adjacency(did, wid, eid, self.feat_id)  # 包含 api 请求

        timestamp = datetime.now().strftime('%H_%M_%S')
        target_path = os.path.join(macro.SAVE_ROOT, f'face_topo_{sys._getframe(0).f_lineno}_{timestamp}.json')
        with open(target_path, 'w') as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=4)

        self._to_local_coordinates()
        self._build_lookup()

    def _to_local_coordinates(self):
        """
        transform into local coordinate system
        """
        self.origin = np.array(self.plane["origin"])
        self.z_axis = np.array(self.plane["normal"])
        self.x_axis = np.array(self.plane["x"])
        self.y_axis = np.cross(self.plane["normal"], self.plane["x"])
        for item in self.geo_topo["vertices"]:
            old_vec = np.array(item["param"]["Vector"])
            new_vec = old_vec - self.origin
            item["param"]["Vector"] = [np.dot(new_vec, self.x_axis),
                                       np.dot(new_vec, self.y_axis),
                                       np.dot(new_vec, self.z_axis)]

        for item in self.geo_topo["edges"]:
            if item["param"]["type"] == "Circle":
                old_vec = np.array(item["param"]["coordSystem"]["origin"])
                new_vec = old_vec - self.origin
                item["param"]["coordSystem"]["origin"] = [np.dot(new_vec, self.x_axis),
                                                          np.dot(new_vec, self.y_axis),
                                                          np.dot(new_vec, self.z_axis)]

    def _build_lookup(self):
        """
        build a look up table with entity ID as key
        """
        edge_table = {}
        for item in self.geo_topo["edges"]:
            edge_table.update({item["id"]: item})
        self.edge_table = edge_table

        vert_table = {}
        for item in self.geo_topo["vertices"]:
            vert_table.update({item["id"]: item})
        self.vert_table = vert_table

    def _parse_edges_to_loops(self, all_edge_ids):
        """
        sort all edges of a face into loops.
        FIXME: this can be error-prone. bug situation: one vertex connected to 3 edges
        """
        vert2edge = {}
        for edge_id in all_edge_ids:
            item = self.edge_table[edge_id]
            for vert in item["vertices"]:
                if vert not in vert2edge.keys():
                    vert2edge.update({vert: [item["id"]]})
                else:
                    vert2edge[vert].append(item["id"])

        all_loops = []
        unvisited_edges = copy.copy(all_edge_ids)
        while len(unvisited_edges) > 0:
            cur_edge = unvisited_edges[0]
            unvisited_edges.remove(cur_edge)
            loop_edge_ids = [cur_edge]
            if len(self.edge_table[cur_edge]["vertices"]) == 0:  # no corresponding vertices
                pass
            else:
                loop_start_point, cur_end_point = self.edge_table[cur_edge]["vertices"][0], \
                                                  self.edge_table[cur_edge]["vertices"][-1]
                while cur_end_point != loop_start_point:
                    # find next connected edge
                    edges = vert2edge[cur_end_point][:]
                    edges.remove(cur_edge)
                    cur_edge = edges[0]
                    loop_edge_ids.append(cur_edge)
                    unvisited_edges.remove(cur_edge)

                    # find next enc_point
                    points = self.edge_table[cur_edge]["vertices"][:]
                    points.remove(cur_end_point)
                    cur_end_point = points[0]
            all_loops.append(loop_edge_ids)
        return all_loops

    def _parse_edge_to_fusion360_format(self, edge_id):
        """
        parse a edge into fusion360 gallery format. Only support 'Line', 'Circle' and 'Arc'.
        """
        edge_data = self.edge_table[edge_id]
        edge_type = edge_data["param"]["type"]
        if edge_type == "Line":
            start_id, end_id = edge_data["vertices"]
            start_point = xyz_list2dict(self.vert_table[start_id]["param"]["Vector"])
            end_point = xyz_list2dict(self.vert_table[end_id]["param"]["Vector"])
            curve_dict = OrderedDict({"type": "Line3D", "start_point": start_point,
                                      "end_point": end_point, "curve": edge_id})
        elif edge_type == "Circle" and len(edge_data["vertices"]) == 2: # an Arc
            radius = edge_data["param"]["radius"]
            start_id, end_id = edge_data["vertices"]
            start_point = xyz_list2dict(self.vert_table[start_id]["param"]["Vector"])
            end_point = xyz_list2dict(self.vert_table[end_id]["param"]["Vector"])
            center_point = xyz_list2dict(edge_data["param"]["coordSystem"]["origin"])
            normal = xyz_list2dict(edge_data["param"]["coordSystem"]["zAxis"])

            start_vec = np.array(self.vert_table[start_id]["param"]["Vector"]) - \
                        np.array(edge_data["param"]["coordSystem"]["origin"])
            end_vec = np.array(self.vert_table[end_id]["param"]["Vector"]) - \
                      np.array(edge_data["param"]["coordSystem"]["origin"])
            start_vec = start_vec / np.linalg.norm(start_vec)
            end_vec = end_vec / np.linalg.norm(end_vec)

            start_angle = angle_from_vector_to_x(start_vec)
            end_angle = angle_from_vector_to_x(end_vec)
            # keep it counter-clockwise first
            if start_angle > end_angle:
                start_angle, end_angle = end_angle, start_angle
                start_vec, end_vec = end_vec, start_vec
            sweep_angle = abs(start_angle - end_angle)

            # # decide direction arc by curve length
            # edge_len = self.client.eval_curveLength(self.did, self.wid, self.eid, edge_id)
            # _len = sweep_angle * radius
            # _len_other = (2 * np.pi - sweep_angle) * radius
            # if abs(edge_len - _len) > abs(edge_len - _len_other):
            #     sweep_angle = 2 * np.pi - sweep_angle
            #     start_vec = end_vec

            # decide direction by middle point
            midpoint = self.client.eval_curve_midpoint(self.did, self.wid, self.eid, edge_id)
            mid_vec = np.array(midpoint) - self.origin
            mid_vec = np.array([np.dot(mid_vec, self.x_axis), np.dot(mid_vec, self.y_axis), np.dot(mid_vec, self.z_axis)])
            mid_vec = mid_vec - np.array(edge_data["param"]["coordSystem"]["origin"])
            mid_vec = mid_vec / np.linalg.norm(mid_vec)
            mid_angle_real = angle_from_vector_to_x(mid_vec)
            mid_angle_now = (start_angle + end_angle) / 2
            if round(mid_angle_real, 3) != round(mid_angle_now, 3):
                sweep_angle = 2 * np.pi - sweep_angle
                start_vec = end_vec

            ref_vec_dict = xyz_list2dict(list(start_vec))
            curve_dict = OrderedDict({"type": "Arc3D", "start_point": start_point, "end_point": end_point,
                          "center_point": center_point, "radius": radius, "normal": normal,
                          "start_angle": 0.0, "end_angle": sweep_angle, "reference_vector": ref_vec_dict,
                          "curve": edge_id})

        elif edge_type == "Circle" and len(edge_data["vertices"]) < 2:
            # NOTE: treat the circle with only one connected vertex as a full circle
            radius = edge_data["param"]["radius"]
            center_point = xyz_list2dict(edge_data["param"]["coordSystem"]["origin"])
            normal = xyz_list2dict(edge_data["param"]["coordSystem"]["zAxis"])
            curve_dict = OrderedDict({"type": "Circle3D", "center_point": center_point, "radius": radius, "normal": normal,
                          "curve": edge_id})
        else:
            raise NotImplementedError(edge_type, edge_data["vertices"])
        return curve_dict

    def parse_to_fusion360_format(self):
        """
        parse sketch feature into fusion360 gallery format
        """
        name = self.feat_name

        # transform & reference plane
        transform_dict = {"origin": xyz_list2dict(self.plane["origin"]),
                          "z_axis": xyz_list2dict(self.plane["normal"]),
                          "x_axis": xyz_list2dict(self.plane["x"]),
                          "y_axis": xyz_list2dict(list(np.cross(self.plane["normal"], self.plane["x"])))}
        ref_plane_dict = {}

        # profiles
        profiles_dict = {}
        for item in self.geo_topo['faces']:
            # profile level
            profile_id = item['id']  # 目标 JGC
            all_edge_ids = item['edges']
            edge_ids_per_loop = self._parse_edges_to_loops(all_edge_ids)
            all_loops = []
            for loop in edge_ids_per_loop:
                curves = [self._parse_edge_to_fusion360_format(edge_id) for edge_id in loop]
                loop_dict = {"is_outer": True, "profile_curves": curves}
                all_loops.append(loop_dict)
            profiles_dict.update({profile_id: {"loops": all_loops, "properties": {}}})

        entity_dict = {"name": name, "type": "Sketch", "profiles": profiles_dict,
                       "transform": transform_dict, "reference_plane": ref_plane_dict}
        return entity_dict


def process_one(link, is_load_ofs):
    """
    data_id: model number
    link: model link on onShape
    save_dir: parsed data save dir
    is_load_ofs: 是否直接从文件读取序列，而不是通过 onshape_api request?
    """
    # create instance of the OnShape client; change key to test on another stack
    onshape_client = MyClient()

    v_list = link.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

    # filter data that use operations other than sketch + extrude
    ofs_json_file = os.path.join(macro.SAVE_ROOT, 'orig_ofs.json')
    if is_load_ofs:
        with open(ofs_json_file, 'r') as f:
            orig_ofs = json.load(f)

    else:
        orig_ofs = onshape_client.get_features(did, wid, eid).json()
        try:
            with open(ofs_json_file, 'w') as f:
                json.dump(orig_ofs, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(e)
            exit(0)

    parser = FeatureListParser(onshape_client, did, wid, eid, orig_ofs)  # 构造函数不进行 api 请求
    result = parser.parse()  # 包含 api 请求

    if len(result["sequence"]) < 2:
        return 0

    return result


def test_read():
    json_file = r'E:\document\DeeplearningIdea\multi_cmd_seq_gen\four_type_ofs\sketch_topology_581_093424.json'
    with open(json_file, 'r') as f:
        res_msg = json.load(f)

    res_msg = res_msg['result']['message']['value']

    topo = {}
    for item in res_msg:
        k_str = item['message']['key']['message']['value']  # faces, edges
        v_item = item['message']['value']['message']['value']
        outer_list = []

        for item_x in v_item:
            v_item_x = item_x['message']['value']
            geo_dict = {}

            for item_y in v_item_x:
                k = item_y['message']['key']['message']['value']  # id, edges/vertices
                v_msg = item_y['message']['value']

                if k == 'param':
                    if k_str == 'faces':
                        v = MyClient.parse_face_msg(v_msg)[0]

                    elif k_str == 'edges':
                        v = MyClient.parse_edge_msg(v_msg)[0]

                    elif k_str == 'vertices':
                        v = MyClient.parse_vertex_msg(v_msg)[0]

                    else:
                        raise ValueError

                elif isinstance(v_msg['message']['value'], list):
                    v = [a['message']['value'] for a in v_msg['message']['value']]

                else:
                    v = v_msg['message']['value']

                geo_dict.update({k: v})
            outer_list.append(geo_dict)
        topo.update({k_str: outer_list})

    res_topo = topo
    return topo


def test():
    process_one(macro.URL, False)
    print('trans finished')








