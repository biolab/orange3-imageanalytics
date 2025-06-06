from typing import Callable

import os
import tempfile
from contextlib import contextmanager

import filelock
import requests


class classproperty(property):
    def __get__(self, instance, owner=None):
        return self.fget(owner)


@contextmanager
def atomic_update(path: str, mode="wb"):
    """
    A context manager implementing an 'atomic' file update (replace).

    The manager yields an open file descriptor (write mode) to a unique
    temporary file created within the same directory as `path`.
    Once the context is exited, the temporary is moved atomically into
    place using `os.replace`.

    Note
    ----
    The executing user must have the permissions to create files in
    the target directory.
    """
    dirname, basename = os.path.dirname(path), os.path.basename(path)
    fd, fn = tempfile.mkstemp(dir=dirname, suffix=f"{basename}-")
    try:
        with open(fd, mode) as f:
            yield f
            os.chmod(fn, 0o644)
    except BaseException:
        try:
            os.unlink(fn)
        except OSError:
            pass
        raise
    else:
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            stat = None
        if stat is not None:
            try:
                os.chmod(fn, stat.st_mode)
            except OSError:
                pass
        os.replace(fn, path)


def parse_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def download_url_to_file(
    url: str, dst: str, *,
    force = False,
    progress_callback: Callable[[int, int], None] | None = None
) -> None:
    """
    Download content of `url` to `dst` local fs path.
    """
    size = -1
    count = 0
    with filelock.FileLock(dst + ".lock"):
        if not os.path.isfile(dst) or force:
            with atomic_update(dst) as f:
                with requests.get(url, stream=True) as s:
                    # Raise error on non 200 OK otherwise 'dst' will be created
                    s.raise_for_status()
                    clength = s.headers.get("content-length", "")
                    clength = parse_int(clength)
                    if clength is not None:
                        size = clength
                    for chunk in s.iter_content(chunk_size=2 ** 16):
                        f.write(chunk)
                        count += len(chunk)
                        if progress_callback:
                            progress_callback(count, size)
