from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLayout
from Orange.data import Table
from Orange.widgets.gui import widgetBox, widgetLabel, comboBox, auto_commit
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget, Default
from requests.exceptions import ConnectionError

from orangecontrib.imageanalytics.image_embedder import ImageEmbedder


class _Input:
    IMAGES = "Images"


class _Output:
    EMBEDDINGS = "Embeddings"
    MISSING_IMAGES = "Missing Images"


class OWImageNetEmbedding(OWWidget):
    name = "ImageNet Embedding"
    description = "Image embedding through deep neural network from ImageNet."
    icon = "icons/ImageNetEmbedding.svg"
    priority = 150

    want_main_area = False
    _auto_apply = Setting(default=True)

    inputs = [(_Input.IMAGES, Table, "set_data")]
    outputs = [
        (_Output.EMBEDDINGS, Table, Default),
        (_Output.MISSING_IMAGES, Table)
    ]

    cb_image_attr_current_id = Setting(default=0)
    _NO_DATA_INFO_TEXT = "No data on input."

    def __init__(self):
        super().__init__()
        self._image_attributes = None
        self._input_data_table = None

        self._setup_layout()

        self._image_embedder = ImageEmbedder(server_url='127.0.0.1')
        connection_info = self._image_embedder.is_connected_to_server()
        self._set_server_info(connection_info)

    def _setup_layout(self):
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        widget_box = widgetBox(self.controlArea, "Info")
        self.input_data_info = widgetLabel(widget_box, self._NO_DATA_INFO_TEXT)

        widget_box = widgetBox(self.controlArea, "Settings")
        self.cb_image_attr = comboBox(
            widget=widget_box,
            master=self,
            value="cb_image_attr_current_id",
            label="Image attribute:",
            orientation=Qt.Horizontal,
            callback=self._cb_image_attr_changed
        )

        auto_commit(
            widget=self.controlArea,
            master=self,
            value="_auto_apply",
            label="Apply",
            checkbox_label="Auto Apply",
            commit=self.commit
        )

    def set_data(self, data):
        if data is None:
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.MISSING_IMAGES, None)
            self.input_data_info.setText(self._NO_DATA_INFO_TEXT)
            return

        self._image_attributes = self._filter_image_attributes(data)
        if not self._image_attributes:
            input_data_info_text = (
                "Data with {:d} instances, but without image attributes."
                .format(len(data)))
            input_data_info_text.format(input_data_info_text)
            self.input_data_info.setText(input_data_info_text)
            self._input_data_table = None
            return

        if not self.cb_image_attr_current_id < len(self._image_attributes):
            self.cb_image_attr_current_id = 0

        self.cb_image_attr.setModel(VariableListModel(self._image_attributes))
        self.cb_image_attr.setCurrentIndex(self.cb_image_attr_current_id)

        self._input_data_table = data
        input_data_info_text = "Data with {:d} instances.".format(len(data))
        self.input_data_info.setText(input_data_info_text)

        self._cb_image_attr_changed()

    @staticmethod
    def _filter_image_attributes(data):
        metas = data.domain.metas
        return [m for m in metas if m.attributes.get("type") == "image"]

    def _cb_image_attr_changed(self):
        if self._auto_apply:
            self.commit()

    def commit(self):
        file_paths_attr = self._image_attributes[self.cb_image_attr_current_id]
        file_paths = self._input_data_table[:, file_paths_attr].metas.flatten()

        with self.progressBar(len(file_paths)) as progress:
            try:
                embeddings = self._image_embedder(
                    file_paths,
                    image_processed_callback=lambda: progress.advance()
                )
            except ConnectionError:
                self.send(_Output.EMBEDDINGS, None)
                self.send(_Output.MISSING_IMAGES, None)
                self._set_server_info(connected=False)
                return

        print(embeddings)
        # todo: construct Orange table and send output signals

    def _set_server_info(self, connected):
        self.clear_messages()
        if connected:
            self.information("Connected to server.")
        else:
            self.warning("Not connected to server.")

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self._image_embedder.__exit__(None, None, None)


if __name__ == '__main__':
    import sys
    from AnyQt.QtWidgets import QApplication
    app = QApplication(sys.argv)
    widget = OWImageNetEmbedding()
    widget.show()
    app.exec()
    widget.saveSettings()
