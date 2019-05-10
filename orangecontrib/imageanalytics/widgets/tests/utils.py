import os
import numpy as np

from Orange.data import StringVariable, Table, Domain


def load_images():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "test_images")
    files = [[x] for x in os.listdir(path)]

    image_var = StringVariable(name="Images")
    image_var.attributes["type"] = "image"
    image_var.attributes["origin"] = path

    table = Table(Domain([], [], [image_var]),
                  np.empty((len(files), 0), dtype=float), None, files)

    return table
