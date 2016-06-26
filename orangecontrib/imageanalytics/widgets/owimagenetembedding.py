"""
ImageNet Embedding
------------------
Embeds images listed in input data table using the output of the
image embedding server.
"""

# todo how to handle dirty flag on commit (checkbox auto commit)
# todo check when dead server
# todo recheck the server (should there be another button, where?)
# todo handle images from the network (not in local files)
# todo add image attribute (like in view images)

import numpy as np

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from orangecontrib.imageanalytics.embeddimage import ImageProfiler
from PyQt4 import QtGui
from PyQt4.QtGui import QDesktopServices
from PyQt4.QtCore import Qt, QUrl


class OWImageNetEmbedding(widget.OWWidget):
    name = "ImageNet Embedding"
    description = "Image profiling through deep network from ImageNet."
    icon = "icons/ImageNetEmbedding.svg"
    priority = 150

    inputs = [("Data", Table, "set_data")]
    outputs = [("Embeddings", Table, widget.Default)]
    auto_commit = Setting(True)
    token = Setting("")

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.file_att = None  # attribute with image file names
        self.origin = None  # origin of image files

        box = gui.widgetBox(self.controlArea, "Info")
        self.info_a = gui.widgetLabel(box, "No data on input.")
        self.info_b = gui.widgetLabel(box, "")

        box = gui.widgetBox(self.controlArea, "Server Token")
        gui.lineEdit(box, self, "token", "Token: ",
                     controlWidth=250, orientation=Qt.Horizontal,
                     enterPlaceholder=True, callback=self.token_name_changed)

        gui.button(
            box, self, "Get Token",
            callback=self.open_server_token_page,
            autoDefault=False
        )

        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)

        self.commit_box = gui.auto_commit(
            self.controlArea, self, "auto_commit", label="Commit",
            checkbox_label="Commit on change"
        )
        self.commit_box.setEnabled(False)

        self.profiler = ImageProfiler(token=self.token)
        if self.token:
            self.profiler.set_token(self.token)
        self.set_info()

    def set_info(self):
        if self.profiler.token:
            if self.profiler.server:
                self.info_b.setText(
                    "Connected to server. Credit for {} "
                    "image{}.".format(self.profiler.coins,
                                      "s" if self.profiler.coins > 1 else ""))
                if self.profiler.coins > 0:
                    self.commit_box.setEnabled(True)
                    self.warning(0, "")
            else:
                self.info_b.setText("Connection with server not established.")
        else:
            self.info_b.setText("Please enter valid server token.")

    def open_server_token_page(self):
        url = QUrl(self.profiler.server + "get_token")
        QDesktopServices.openUrl(url)

    def set_data(self, data):
        if data is None:
            self.send("Embeddings", None)
            self.info_a.setText("No data on input.")
            return

        self.info_a.setText("Data with %d instances." % len(data))
        self.data = data
        # todo check for image attributes

        atts = [a for a in data.domain.metas if
                a.attributes.get("type") == "image"]
        if not atts:
            self.warning(text="Input data has no image attributes.")
            self.info_a.setText("Data (%d instances) without image "
                                "attributes." % len(data))
            self.data = None
            return
        self.file_att = atts[0]  # todo handle a set of image attributes
        self.origin = self.file_att.attributes.get("origin", "")
        if self.auto_commit:
            self.commit()

    def token_name_changed(self):
        self.profiler.set_token(self.token)
        if self.profiler.token:  # token is valid
            if self.auto_commit:
                self.commit()
        self.set_info()

    def commit(self):
        if len(self.data) > self.profiler.coins:
            self.warning(0, "Not enough credit to process images.")
            return
        with self.progressBar(len(self.data)) as progress:
            if not self.profiler.server:
                self.info_b.setText("Connection with server not established.")
                self.send("Embeddings", None)
                return
            xp = []
            for d in self.data:
                filename = self.origin + "/" + str(d[self.file_att])
                ps = self.profiler(filename)
                xp.append(ps)
                progress.advance()
        self.profiler.dump_history()

        xp = np.array(xp)
        x = np.hstack((self.data.X, xp))
        atts = [ContinuousVariable("n%d" % i) for i in
                range(xp.shape[1])]
        domain = Domain(list(self.data.domain.attributes) + atts,
                        self.data.domain.class_vars,
                        self.data.domain.metas)
        embeddings = Table(domain, x, self.data.Y, self.data.metas)
        self.send("Embeddings", embeddings)
        self.set_info()


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    app = QApplication(sys.argv)
    ow = OWImageNetEmbedding()
    origin = "/Users/blaz/Desktop/images/dicty"
    data = Table(origin + "/small.tab")
    data.domain["image"].attributes.update({"origin": origin})
    ow.set_data(data)
    ow.show()
    app.exec()
    ow.saveSettings()
