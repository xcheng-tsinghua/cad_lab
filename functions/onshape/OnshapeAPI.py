import json
import os
import random
import string
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

