import logging
from types import SimpleNamespace
from typing import Optional

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLayout, QPushButton, QStyle

from Orange.data import Table, Variable
from Orange.misc.utils.embedder_utils import EmbeddingConnectionError
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from orangecontrib.imageanalytics.image_embedder import \
    MODELS as EMBEDDERS_INFO
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
from orangecontrib.imageanalytics.utils.image_utils import filter_image_attributes


class Result(SimpleNamespace):
    embedding: Optional[Table] = None
    skip_images: Optional[Table] = None
    num_skipped: int = None


def run_embedding(
    images: Table,
    file_paths_attr: Variable,
    embedder_name: str,
    state: TaskState,
) -> Result:
    """
    Run the embedding process

    Parameters
    ----------
    images
        Data table with images to embed.
    file_paths_attr
        The column of the table with images.
    embedder_name
        The name of selected embedder.
    state
        State object used for controlling and progress.

    Returns
    -------
    The object that holds embedded images, skipped images, and number
    of skipped images.
    """
    embedder = ImageEmbedder(model=embedder_name)

    def callback(s):
        if state.is_interruption_requested():
            raise Exception()
        state.set_progress_value(s * 100)

    try:
        emb, skip, n_skip = embedder(images, col=file_paths_attr, callback=callback)
    except EmbeddingConnectionError:
        state.set_partial_result("squeezenet")
        embedder = ImageEmbedder(model="squeezenet")
        emb, skip, n_skip = embedder(images, col=file_paths_attr, callback=callback)

    return Result(embedding=emb, skip_images=skip, num_skipped=n_skip)


class OWImageEmbedding(OWWidget, ConcurrentWidgetMixin):
    name = "Image Embedding"
    description = "Image embedding through deep neural networks."
    keywords = "image embedding, embedding, image"
    icon = "icons/ImageEmbedding.svg"
    priority = 150

    want_main_area = False
    buttons_area_orientation = Qt.Vertical
    _auto_apply = Setting(default=True)

    class Inputs:
        images = Input("Images", Table)

    class Outputs:
        embeddings = Output("Embeddings", Table, default=True)
        skipped_images = Output("Skipped Images", Table)

    class Warning(OWWidget.Warning):
        switched_local_embedder = Msg(
            "No internet connection: switched to local embedder"
        )
        no_image_attribute = Msg(
            "Please provide data with an image attribute."
        )
        images_skipped = Msg("{} images are skipped.")

    class Error(OWWidget.Error):
        unexpected_error = Msg("Embedding error: {}")

    cb_image_attr_current_id = Setting(default=0)
    cb_embedder_current_id = Setting(default=0)
    custom_server = Setting(default=False)
    server_url = Setting(default="")

    _previous_attr_id = None
    _previous_embedder_id = None

    _NO_DATA_INFO_TEXT = "No data on input."

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.embedders = sorted(
            list(EMBEDDERS_INFO), key=lambda k: EMBEDDERS_INFO[k]["order"]
        )
        self._image_attributes = None
        self._input_data = None
        self._log = logging.getLogger(__name__)
        self._task = None
        self._setup_layout()

    def _setup_layout(self):
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        widget_box = gui.widgetBox(self.controlArea, "Settings")
        self.cb_image_attr = gui.comboBox(
            widget=widget_box,
            master=self,
            value="cb_image_attr_current_id",
            label="Image attribute:",
            orientation=Qt.Horizontal,
            callback=self._cb_image_attr_changed,
        )

        self.cb_embedder = gui.comboBox(
            widget=widget_box,
            master=self,
            value="cb_embedder_current_id",
            label="Embedder:",
            orientation=Qt.Horizontal,
            callback=self._cb_embedder_changed,
        )
        names = [
            EMBEDDERS_INFO[e]["name"]
            + (" (local)" if EMBEDDERS_INFO[e].get("is_local") else "")
            for e in self.embedders
        ]
        self.cb_embedder.setModel(VariableListModel(names))
        if not self.cb_embedder_current_id < len(self.embedders):
            self.cb_embedder_current_id = 0
        self.cb_embedder.setCurrentIndex(self.cb_embedder_current_id)

        current_embedder = self.embedders[self.cb_embedder_current_id]
        self.embedder_info = gui.widgetLabel(
            widget_box, EMBEDDERS_INFO[current_embedder]["description"]
        )

        advanced_box = gui.widgetBox(self.controlArea, 'Custom server')
        self.use_custom_server_cb = gui.checkBox(
            advanced_box, self, 'custom_server', "Use private server",
            callback=self.changed_server_url)
        self.server_url_box = gui.lineEdit(
            advanced_box, self, 'server_url', label="Server URL: ",
            tooltip='The url of the server for calculating image embeddings',
            orientation=Qt.Horizontal, callback=self.changed_server_url)

        self.auto_commit_widget = gui.auto_commit(
            widget=self.buttonsArea,
            master=self,
            value="_auto_apply",
            label="Apply",
        )

        self.cancel_button = QPushButton(
            "Cancel",
            icon=self.style().standardIcon(QStyle.SP_DialogCancelButton),
        )
        self.cancel_button.clicked.connect(self.cancel)
        self.buttonsArea.layout().addWidget(self.cancel_button)
        self.cancel_button.setDisabled(True)

    @Inputs.images
    def set_data(self, data):
        self.Warning.clear()
        self.set_input_data_summary(data)
        self.clear_outputs()

        if not data:
            self._input_data = None
            return

        self._image_attributes = filter_image_attributes(data)
        if not self.cb_image_attr_current_id < len(self._image_attributes):
            self.cb_image_attr_current_id = 0

        self.cb_image_attr.setModel(VariableListModel(self._image_attributes))
        self.cb_image_attr.setCurrentIndex(self.cb_image_attr_current_id)

        if not self._image_attributes:
            self._input_data = None
            self.Warning.no_image_attribute()
            self.clear_outputs()
            return

        self._input_data = data
        self._previous_attr_id = self.cb_image_attr_current_id
        self._previous_embedder_id = self.cb_embedder_current_id

        self.commit.now()

    def _cb_image_attr_changed(self):
        self._cb_changed()

    def changed_server_url(self):
        self._init_server_connection()
        self.server_url_box.setDisabled(not self.custom_server)

    def _cb_embedder_changed(self):
        self.Warning.switched_local_embedder.clear()
        current_embedder = self.embedders[self.cb_embedder_current_id]
        self.embedder_info.setText(
            EMBEDDERS_INFO[current_embedder]["description"]
        )
        self._cb_changed()

    def _cb_changed(self):
        if (
            self._previous_embedder_id != self.cb_embedder_current_id
            or self._previous_attr_id != self.cb_image_attr_current_id
        ):
            # recompute embeddings only when selected value in dropdown changes
            self._previous_embedder_id = self.cb_embedder_current_id
            self._previous_attr_id = self.cb_image_attr_current_id
            self.cancel()
            self.commit.deferred()

    @gui.deferred
    def commit(self):
        if not self._image_attributes or self._input_data is None:
            self.clear_outputs()
            return

        self.cancel_button.setDisabled(False)

        embedder_name = self.embedders[self.cb_embedder_current_id]
        image_attribute = self._image_attributes[self.cb_image_attr_current_id]
        self.start(
            run_embedding, self._input_data, image_attribute, embedder_name
        )
        self.Error.unexpected_error.clear()

    def on_done(self, result: Result) -> None:
        """
        Invoked when task is done.

        Parameters
        ----------
        result
            Embedding results.
        """
        self.cancel_button.setDisabled(True)
        assert len(self._input_data) == len(result.embedding or []) + len(
            result.skip_images or []
        )
        self._send_output_signals(result)

    def on_partial_result(self, result: str) -> None:
        self._switch_to_local_embedder()

    def on_exception(self, ex: Exception) -> None:
        """
        When an exception occurs during the calculation.

        Parameters
        ----------
        ex
            Exception occurred during the embedding.
        """
        log = logging.getLogger(__name__)
        log.debug(ex, exc_info=ex)
        self.cancel_button.setDisabled(True)
        self.Error.unexpected_error(str(ex))
        self.clear_outputs()

    def cancel(self):
        self.cancel_button.setDisabled(True)
        super().cancel()

    def _switch_to_local_embedder(self):
        self.Warning.switched_local_embedder()
        self.cb_embedder_current_id = self.embedders.index("squeezenet")

    def _send_output_signals(self, result: Result) -> None:
        self.Warning.images_skipped.clear()
        self.Outputs.embeddings.send(result.embedding)
        self.Outputs.skipped_images.send(result.skip_images)
        if result.num_skipped != 0:
            self.Warning.images_skipped(result.num_skipped)
        self.set_output_data_summary(result.embedding, result.skip_images)

    def clear_outputs(self):
        self._send_output_signals(
            Result(embedding=None, skpped_images=None, num_skipped=0)
        )

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWImageEmbedding).run(
        Table("https://datasets.biolab.si/core/bone-healing.xlsx")
    )
