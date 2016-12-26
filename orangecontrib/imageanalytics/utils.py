import hashlib
import pickle
import re

_HOSTNAME_REGEX = re.compile(r'^https?://(?P<hostname>[\wd.]+)\b')


def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def get_hostname(text):
    url_match = re.match(_HOSTNAME_REGEX, text)
    return url_match.group('hostname')


def md5_hash(bytes_):
    md5 = hashlib.md5()
    md5.update(bytes_)
    return md5.hexdigest()
