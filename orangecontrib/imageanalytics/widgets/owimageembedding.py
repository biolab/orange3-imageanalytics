import logging
from types import SimpleNamespace
from typing import Optional

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QPushButton, QStyle
from AnyQt.QtWidgets import QFormLayout

from orangewidget.utils.combobox import ComboBox

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
    if embedder.is_local_embedder():
        model_cls = embedder._model_settings["model"]
        if not model_cls.is_cached():
            def callback(count, total):
                if state.is_interruption_requested():
                    raise Exception()
                if total > 0:
                    state.set_progress_value(count / total * 100)
            state.set_status("Downloading model. Please hold.")
            try:
                model_cls.download_from_hf(progress_callback=callback)
            finally:
                state.set_status("")
                state.set_progress_value(0)
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
    resizing_enabled = False
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

    settings_version = 2
    cb_image_attr_current_id = Setting(default=0)

    current_embedder: str = Setting("inceptionnext_atto-local")
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
        form = QFormLayout(
            spacing=8,
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )
        gui.widgetBox(self.controlArea, "Settings", orientation=form)

        self.cb_image_attr = gui.comboBox(
            widget=None,
            master=self,
            value="cb_image_attr_current_id",
            label="Image attribute:",
            orientation=Qt.Horizontal,
            callback=self._cb_image_attr_changed,
        )
        self.cb_embedder = ComboBox()
        names = [
            (e, EMBEDDERS_INFO[e]["name"]
            + (" (Remote)" if not EMBEDDERS_INFO[e].get("is_local") else ""))
            for e in self.embedders
        ]
        for key, name in names:
            self.cb_embedder.addItem(name, userData=key)

        index = self.cb_embedder.findData(self.current_embedder)
        if index == -1:
            index = 0
        self.cb_embedder.setCurrentIndex(index)
        current_embedder = self.cb_embedder.currentData()
        self.cb_embedder.currentIndexChanged.connect(self._cb_embedder_changed)
        self.embedder_info = gui.widgetLabel(
            None, EMBEDDERS_INFO[current_embedder]["description"]
        )
        form.addRow("Image attribute:", self.cb_image_attr)
        form.addRow("Embedder:", self.cb_embedder)
        form.setWidget(2, QFormLayout.ItemRole.SpanningRole, self.embedder_info)

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

    def set_input_data_summary(self, data):
        if data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(
                str(len(data)), f"Data have {len(data)} instances"
            )

    def set_output_data_summary(self, data_emb, data_skip):
        if data_emb is None and data_skip is None:
            self.info.set_output_summary(self.info.NoOutput)
        else:
            success = 0 if data_emb is None else len(data_emb)
            skip = 0 if data_skip is None else len(data_skip)
            self.info.set_output_summary(
                f"{success}",
                f"{success} images successfully embedded ,\n"
                f"{skip} images skipped.",
            )

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
        self.commit.now()

    def _cb_image_attr_changed(self):
        if self._previous_attr_id != self.cb_image_attr_current_id:
            # recompute embeddings only when selected value in dropdown changes
            self._previous_attr_id = self.cb_image_attr_current_id
            self.cancel()
            self.commit.deferred()

    def _cb_embedder_changed(self):
        key = self.cb_embedder.currentData()
        self.set_current_embedder(key)

    def set_current_embedder(self, key):
        """Set current selected embedder

        `key` must key into EMBEDDERS_INFO
        """
        index = self.cb_embedder.findData(key)
        assert index != -1
        if self.current_embedder != key:
            self.cb_embedder.setCurrentIndex(index)
            self.current_embedder = key
            self.embedder_info.setText(
                EMBEDDERS_INFO[key]["description"]
            )
            self.cancel()
            self.commit.deferred()
            self.Warning.switched_local_embedder.clear()

    @gui.deferred
    def commit(self):
        if not self._image_attributes or self._input_data is None:
            self.clear_outputs()
            return

        self.cancel_button.setDisabled(False)

        embedder_name = self.current_embedder
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
        self.Error.unexpected_error(str(ex), exc_info=ex)
        self.clear_outputs()

    def cancel(self):
        self.cancel_button.setDisabled(True)
        super().cancel()

    def _switch_to_local_embedder(self):
        self.Warning.switched_local_embedder()
        self.set_current_embedder("squeezenet")

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

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is None or version < 2:
            cb_embedder_current_id = settings.pop("cb_embedder_current_id", 0)
            current_embedder = [
                "inception-v3",
                "squeezenet",
                "vgg16",
                "vgg19",
                "painters",
                "deeploc",
                "openface",
            ][cb_embedder_current_id]
            settings["current_embedder"] = current_embedder


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWImageEmbedding).run(
        Table("https://datasets.biolab.si/core/bone-healing.xlsx")
    )
