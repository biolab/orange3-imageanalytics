import os
from typing import List, Union, Optional
from urllib.parse import urlparse, urljoin

from Orange.data import (
    Table,
    Variable,
    StringVariable,
    DiscreteVariable,
    Instance,
    Unknown,
)


def filter_image_attributes(data: Table) -> List[Variable]:
    """
    Filter out attributes which can potentially be an image attributes. The
    selection includes all String and Categorical attributes from meta part of
    the table. Put attributes with "type" == "image" first - they are more likely
    to be image attributes.

    Parameters
    ----------
    data
        Table with data

    Returns
    -------
    List of variables that can be image attributes
    """
    m = data.domain.metas
    m = [a for a in m if isinstance(a, (StringVariable, DiscreteVariable))]
    return sorted(m, key=lambda a: a.attributes.get("type") == "image", reverse=True)


def extract_paths(
    data: Table, column: Union[DiscreteVariable, StringVariable]
) -> List[Optional[str]]:
    """
    Extract image/file paths from datatable column. If column has an origin attribute
    it will be added as a prefix to the path otherwise just values of column will be
    returned.

    Parameters
    ----------
    data
        Table that have column with file paths/URL in metas
    column
        Variable which is a path attribute. Must be string or categorical

    Returns
    -------
    List of file paths or URLs
    """
    return [extract_image_path(inst, column) for inst in data]


def extract_image_path(
    instance: Instance, attribute: Union[DiscreteVariable, StringVariable]
) -> Optional[str]:
    """
    Extract image/file path from instance's attribute. If attribute has an origin
    attribute it will be added as a prefix to the path otherwise just values of column
    will be returned.

    Parameters
    ----------
    instance
        Instance that have column with file paths/URL in metas
    attribute
        Variable which is a path attribute. Must be string or categorical

    Returns
    -------
    File paths or URL
    """
    file_path = instance[attribute].value
    if file_path == "" or file_path is Unknown:
        return None

    origin = attribute.attributes.get("origin", "")
    if (
        urlparse(origin).scheme in ("http", "https", "ftp", "data")
        and origin[-1] != "/"
    ):
        origin += "/"

    urlparts = urlparse(file_path)
    if urlparts.scheme not in ("http", "https", "ftp", "data"):
        if urlparse(origin).scheme in ("http", "https", "ftp", "data"):
            file_path = urljoin(origin, file_path)
        else:
            file_path = os.path.join(origin, file_path)
    return file_path
