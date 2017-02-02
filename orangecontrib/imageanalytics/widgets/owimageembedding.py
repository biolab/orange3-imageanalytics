import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLayout
from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.gui import widgetBox, widgetLabel, comboBox, auto_commit
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget, Default

from orangecontrib.imageanalytics.image_embedder import ImageEmbedder


class _Input:
    IMAGES = 'Images'


class _Output:
    EMBEDDINGS = 'Embeddings'
    SKIPPED_IMAGES = 'Skipped Images'


class OWImageEmbedding(OWWidget):
    # todo: implement embedding in a non-blocking manner
    # todo: implement stop running task action
    name = "Image Embedding"
    description = "Image embedding through deep neural networks."
    icon = "icons/ImageEmbedding.svg"
    priority = 150

    want_main_area = False
    _auto_apply = Setting(default=True)

    inputs = [(_Input.IMAGES, Table, 'set_data')]
    outputs = [
        (_Output.EMBEDDINGS, Table, Default),
        (_Output.SKIPPED_IMAGES, Table)
    ]

    cb_image_attr_current_id = Setting(default=0)
    _NO_DATA_INFO_TEXT = "No data on input."

    def __init__(self):
        super().__init__()
        self._image_attributes = None
        self._input_data = None

        self._setup_layout()

        self._image_embedder = ImageEmbedder(
            model='inception-v3',
            layer='penultimate',
        )
        self._set_server_info(
            self._image_embedder.is_connected_to_server()
        )

    def _setup_layout(self):
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        widget_box = widgetBox(self.controlArea, 'Info')
        self.input_data_info = widgetLabel(widget_box, self._NO_DATA_INFO_TEXT)
        self.connection_info = widgetLabel(widget_box, "")

        widget_box = widgetBox(self.controlArea, 'Settings')
        self.cb_image_attr = comboBox(
            widget=widget_box,
            master=self,
            value='cb_image_attr_current_id',
            label='Image attribute:',
            orientation=Qt.Horizontal,
            callback=self._cb_image_attr_changed
        )

        self.auto_commit_widget = auto_commit(
            widget=self.controlArea,
            master=self,
            value='_auto_apply',
            label='Apply',
            checkbox_label='Auto Apply',
            commit=self.commit
        )

    def set_data(self, data):
        if data is None:
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            self.input_data_info.setText(self._NO_DATA_INFO_TEXT)
            return

        self._image_attributes = self._filter_image_attributes(data)
        if not self._image_attributes:
            input_data_info_text = (
                "Data with {:d} instances, but without image attributes."
                .format(len(data)))
            input_data_info_text.format(input_data_info_text)
            self.input_data_info.setText(input_data_info_text)
            self._input_data = None
            return

        if not self.cb_image_attr_current_id < len(self._image_attributes):
            self.cb_image_attr_current_id = 0

        self.cb_image_attr.setModel(VariableListModel(self._image_attributes))
        self.cb_image_attr.setCurrentIndex(self.cb_image_attr_current_id)

        self._input_data = data
        input_data_info_text = "Data with {:d} instances.".format(len(data))
        self.input_data_info.setText(input_data_info_text)

        self._cb_image_attr_changed()

    @staticmethod
    def _filter_image_attributes(data):
        metas = data.domain.metas
        return [m for m in metas if m.attributes.get('type') == 'image']

    def _cb_image_attr_changed(self):
        if self._auto_apply:
            self.commit()

    def commit(self):
        if not self._image_attributes or not self._input_data:
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            return

        self._set_server_info(connected=True)
        self.auto_commit_widget.setDisabled(True)

        file_paths_attr = self._image_attributes[self.cb_image_attr_current_id]
        file_paths = self._input_data[:, file_paths_attr].metas.flatten()

        with self.progressBar(len(file_paths)) as progress:
            try:
                embeddings = self._image_embedder(
                    file_paths=file_paths,
                    image_processed_callback=lambda: progress.advance()
                )
            except ConnectionError:
                self.send(_Output.EMBEDDINGS, None)
                self.send(_Output.SKIPPED_IMAGES, None)
                self._set_server_info(connected=False)
                self.auto_commit_widget.setDisabled(False)
                return

        self._send_output_signals(embeddings)
        self.auto_commit_widget.setDisabled(False)

    def _send_output_signals(self, embeddings):
        skipped_images_bool = np.array([x is None for x in embeddings])

        if np.any(skipped_images_bool):
            skipped_images = self._input_data[skipped_images_bool]
            skipped_images = Table(skipped_images)
            skipped_images.ids = self._input_data.ids[skipped_images_bool]
            self.send(_Output.SKIPPED_IMAGES, skipped_images)
        else:
            self.send(_Output.SKIPPED_IMAGES, None)

        embedded_images_bool = np.logical_not(skipped_images_bool)

        if np.any(embedded_images_bool):
            embedded_images = self._input_data[embedded_images_bool]

            embeddings = embeddings[embedded_images_bool]
            embeddings = np.stack(embeddings)

            embedded_images = self._construct_output_data_table(
                embedded_images,
                embeddings
            )
            embedded_images.ids = self._input_data.ids[embedded_images_bool]
            self.send(_Output.EMBEDDINGS, embedded_images)
        else:
            self.send(_Output.EMBEDDINGS, None)

    @staticmethod
    def _construct_output_data_table(embedded_images, embeddings):
        X = np.hstack((embedded_images.X, embeddings))
        Y = embedded_images.Y

        dimensions = range(embeddings.shape[1])
        attributes = [ContinuousVariable('n{:d}'.format(d)) for d in dimensions]
        attributes = list(embedded_images.domain.attributes) + attributes

        domain = Domain(
            attributes=attributes,
            class_vars=embedded_images.domain.class_vars,
            metas=embedded_images.domain.metas
        )

        return Table(domain, X, Y, embedded_images.metas)

    def _set_server_info(self, connected):
        self.clear_messages()
        if connected:
            self.connection_info.setText("Connected to server.")
        else:
            self.connection_info.setText("No connection with server.")
            self.warning("Click Apply to try again.")

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self._image_embedder.__exit__(None, None, None)


if __name__ == '__main__':
    import sys
    from AnyQt.QtWidgets import QApplication
    app = QApplication(sys.argv)
    widget = OWImageEmbedding()
    widget.show()
    app.exec()
    widget.saveSettings()
