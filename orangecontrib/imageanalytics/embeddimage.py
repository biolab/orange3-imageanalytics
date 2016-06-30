import os
import os.path
import numpy as np
import pickle

from PIL import Image
import io
import base64
import hashlib
import Orange.misc.environ
import requests
from requests.exceptions import ConnectionError


class ImageProfiler:
    ENCODING = "utf-8"
    THUMBNAIL_SIZE = (299, 299)
    PICKLE_FILE = Orange.misc.environ.cache_dir() + "/image_embeddings.pkl"

    def __init__(self, server=None, token=None, clear_history=True):
        if os.path.exists(self.PICKLE_FILE) and not clear_history:
            self.history = pickle.load(open(self.PICKLE_FILE, "rb"))
        else:
            self.history = {}

        self.server = server if server else self.get_server_address()
        self.coins = 0
        self.token = ""
        if token:
            self.set_token(token)
        # todo get server address, report if fails (e.g., no network)

    def __enter__(self):
        return self

    def set_token(self, token):
        if self.server and self.check_token_valid(token):
            self.token = token
            self.coins = self.get_coin_count()
        else:
            self.token = ""

    def get_server_address(self):
        # from socket import gethostname
        # if "Air1" in gethostname():
        #     return "http://127.0.0.1:8080/"

        url = "https://raw.githubusercontent.com/biolab/" \
              "orange3-imageanalytics/master/SERVERS.txt"
        try:
            res = requests.get(url)
        except ConnectionError:
            return None
        server = res.text.strip().split("\n")[0]
        if self.check_server_alive(server):
            return server
        else:
            return None

    def check_server_alive(self, server):
        try:
            r = requests.head(server + "/info")
            status = r.status_code
            alive = status == 200
            return alive
        except ConnectionError:
            return False

    @staticmethod
    def compute_hash(obj):
        return hashlib.sha224(obj).hexdigest()

    def get_coin_count(self):
        json = {"token": self.token}
        res = requests.post(self.server + "coin_count", json=json)
        return res.json().get("coins")

    def check_token_valid(self, token):
        json = {"token": token}
        res = requests.post(self.server + "check_token_valid", json=json)
        return res.json().get("valid")

    def __call__(self, image_file_name):
        im = Image.open(image_file_name).convert('RGB')
        im.thumbnail(self.THUMBNAIL_SIZE, Image.ANTIALIAS)
        out = io.BytesIO()
        im.save(out, format="JPEG")
        base64_bytes = base64.b64encode(out.getbuffer())

        h = self.compute_hash(base64_bytes)
        if h in self.history:
            return self.history[h]

        base64_string = base64_bytes.decode(self.ENCODING)
        json = {"image": base64_string, "token": self.token}
        res = requests.post(self.server + "image_profiler", json=json)
        if not res.json()["profile"]:
            return None
        profile = np.array(res.json()["profile"], dtype=np.float16)
        self.history[h] = profile
        self.coins -= 1
        return profile

    def dump_history(self):
        pickle.dump(self.history, open(self.PICKLE_FILE, "wb"), -1)

    def __exit__(self, *_):
        self.dump_history()

if __name__ == "__main__":
    with ImageProfiler(token="3ce7672c-a2ed-4f91-a954-fd34abae5046",
                       clear_history=True) \
            as image_profiler:
        print("Server:", image_profiler.server)
        result = image_profiler("example-image.jpg")
        print(result)
