import logging
import traceback
from types import SimpleNamespace as namespace
import concurrent.futures

import numpy as np
from AnyQt.QtCore import Qt, QThread
from AnyQt.QtCore import pyqtSlot as Slot
from AnyQt.QtTest import QSignalSpy
from AnyQt.QtWidgets import QLayout, QPushButton, QStyle

from Orange.data import Table
from Orange.widgets.gui import hBox
from Orange.widgets.gui import widgetBox, widgetLabel, comboBox, auto_commit
from Orange.widgets.settings import Setting
from Orange.widgets.utils import concurrent as qconcurrent
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import Input, Output, Msg
from Orange.widgets.widget import OWWidget

from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
from orangecontrib.imageanalytics.image_embedder import MODELS as EMBEDDERS_INFO


class OWImageEmbedding(OWWidget):
    name = "Image Embedding"
    description = "Image embedding through deep neural networks."
    keywords = ["embedding", "image", "image embedding"]
    icon = "icons/ImageEmbedding.svg"
    priority = 150

    want_main_area = False
    _auto_apply = Setting(default=True)

    class Inputs:
        images = Input('Images', Table)

    class Outputs:
        embeddings = Output('Embeddings', Table, default=True)
        skipped_images = Output('Skipped Images', Table)

    class Warning(OWWidget.Warning):
        switched_local_embedder = Msg(
            "No internet connection: switched to local embedder")
        no_image_attribute = Msg("Please provide data with an image attribute.")
        images_skipped = Msg("{} images are skipped.")

    cb_image_attr_current_id = Setting(default=0)
    cb_embedder_current_id = Setting(default=0)

    _NO_DATA_INFO_TEXT = "No data on input."

    def __init__(self):
        super().__init__()
        self.embedders = sorted(list(EMBEDDERS_INFO),
                                key=lambda k: EMBEDDERS_INFO[k]['order'])
        self._image_attributes = None
        self._input_data = None
        self._log = logging.getLogger(__name__)
        self._task = None
        self._setup_layout()

    def _setup_layout(self):
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        widget_box = widgetBox(self.controlArea, 'Settings')
        self.cb_image_attr = comboBox(
            widget=widget_box,
            master=self,
            value='cb_image_attr_current_id',
            label='Image attribute:',
            orientation=Qt.Horizontal,
            callback=self._cb_image_attr_changed
        )

        self.cb_embedder = comboBox(
            widget=widget_box,
            master=self,
            value='cb_embedder_current_id',
            label='Embedder:',
            orientation=Qt.Horizontal,
            callback=self._cb_embedder_changed
        )
        names = [EMBEDDERS_INFO[e]['name'] +
                 (" (local)" if EMBEDDERS_INFO[e].get("is_local") else "")
                 for e in self.embedders]
        self.cb_embedder.setModel(VariableListModel(names))
        if not self.cb_embedder_current_id < len(self.embedders):
            self.cb_embedder_current_id = 0
        self.cb_embedder.setCurrentIndex(self.cb_embedder_current_id)

        current_embedder = self.embedders[self.cb_embedder_current_id]
        self.embedder_info = widgetLabel(
            widget_box,
            EMBEDDERS_INFO[current_embedder]['description']
        )

        self.auto_commit_widget = auto_commit(
            widget=self.controlArea,
            master=self,
            value='_auto_apply',
            label='Apply',
            commit=self.commit
        )

        self.cancel_button = QPushButton(
            'Cancel',
            icon=self.style().standardIcon(QStyle.SP_DialogCancelButton),
        )
        self.cancel_button.clicked.connect(self.cancel)
        hbox = hBox(self.controlArea)
        hbox.layout().addWidget(self.cancel_button)
        self.cancel_button.setDisabled(True)

    def set_data_summary(self, data):
        if data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(
                str(len(data)),
                f"Data have {len(data)} instances")

    @Inputs.images
    def set_data(self, data):
        self.Warning.clear()
        self.set_data_summary(data)
        if not data:
            self._input_data = None
            self.clear_outputs()
            return

        self._image_attributes = ImageEmbedder.filter_image_attributes(data)
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

        self.commit()

    def _cb_image_attr_changed(self):
        self.commit()

    def connect(self):
        """
        This function tries to connects to the selected embedder if it is not
        successful due to any server/connection error it switches to the
        local embedder and warns the user about that.
        """
        self.Warning.switched_local_embedder.clear()

        # try to connect to current embedder
        embedder = ImageEmbedder(
            model=self.embedders[self.cb_embedder_current_id],
            layer='penultimate'
        )

        if not embedder.is_local_embedder() and \
            not embedder.is_connected_to_server(use_hyper=False):
            # there is a problem with connecting to the server
            # switching to local embedder
            self.Warning.switched_local_embedder()
            del embedder  # remove current embedder
            self.cb_embedder_current_id = self.embedders.index("squeezenet")
            print(self.embedders[self.cb_embedder_current_id])
            embedder = ImageEmbedder(
                model=self.embedders[self.cb_embedder_current_id],
                layer='penultimate'
            )

        return embedder

    def _cb_embedder_changed(self):
        current_embedder = self.embedders[self.cb_embedder_current_id]
        self.embedder_info.setText(
            EMBEDDERS_INFO[current_embedder]['description'])
        if self._input_data:
            self.commit()

    def commit(self):
        if self._task is not None:
            self.cancel()

        if not self._image_attributes or self._input_data is None:
            self.clear_outputs()
            return

        embedder = self.connect()
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.cancel_button.setDisabled(False)
        self.cb_image_attr.setDisabled(True)
        self.cb_embedder.setDisabled(True)

        file_paths_attr = self._image_attributes[self.cb_image_attr_current_id]
        file_paths = self._input_data[:, file_paths_attr].metas.flatten()

        file_paths_mask = file_paths == file_paths_attr.Unknown
        file_paths_valid = file_paths[~file_paths_mask]

        ticks = iter(np.linspace(0.0, 100.0, file_paths_valid.size))
        set_progress = qconcurrent.methodinvoke(
            self, "__progress_set", (float,))

        def advance(success=True):
            if success:
                set_progress(next(ticks))

        def cancel():
            task.future.cancel()
            task.cancelled = True
            task.embedder.set_canceled(True)

        def run_embedding():
            return embedder(
                self._input_data, col=file_paths_attr,
                image_processed_callback=advance)

        self.auto_commit_widget.setDisabled(True)
        self.progressBarInit()
        self.progressBarSet(0.0)
        self.setBlocking(True)

        f = _executor.submit(run_embedding)
        f.add_done_callback(
            qconcurrent.methodinvoke(self, "__set_results", (object,)))

        task = self._task = namespace(
            file_paths_mask=file_paths_mask,
            file_paths_valid=file_paths_valid,
            file_paths=file_paths,
            embedder=embedder,
            cancelled=False,
            cancel=cancel,
            future=f,
        )
        self._log.debug("Starting embedding task for %i images",
                        file_paths.size)
        return

    @Slot(float)
    def __progress_set(self, value):
        assert self.thread() is QThread.currentThread()
        if self._task is not None:
            self.progressBarSet(value)

    @Slot(object)
    def __set_results(self, f):
        assert self.thread() is QThread.currentThread()
        if self._task is None or self._task.future is not f:
            self._log.info("Reaping stale task")
            return

        assert f.done()

        task, self._task = self._task, None
        self.auto_commit_widget.setDisabled(False)
        self.cancel_button.setDisabled(True)
        self.cb_image_attr.setDisabled(False)
        self.cb_embedder.setDisabled(False)
        self.progressBarFinished()
        self.setBlocking(False)

        try:
            embeddings = f.result()
        except ConnectionError:
            self._log.exception("Error", exc_info=True)
            self.Outputs.embeddings.send(None)
            self.Outputs.skipped_images.send(None)
            return
        except Exception as err:
            self._log.exception("Error", exc_info=True)
            self.error(
                "\n".join(traceback.format_exception_only(type(err), err)))
            self.Outputs.embeddings.send(None)
            self.Outputs.skipped_images.send(None)
            return

        assert self._input_data is not None
        assert len(self._input_data) == len(task.file_paths_mask)

        self._send_output_signals(embeddings)

    def _send_output_signals(self, embeddings):
        self.Warning.images_skipped.clear()
        embedded_images, skipped_images, num_skipped = embeddings
        self.Outputs.embeddings.send(embedded_images)
        self.Outputs.skipped_images.send(skipped_images)
        if num_skipped is not 0:
            self.Warning.images_skipped(num_skipped)

    def clear_outputs(self):
        self.Outputs.embeddings.send(None)
        self.Outputs.skipped_images.send(None)

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()

    def cancel(self):
        if self._task is not None:
            task, self._task = self._task, None
            task.cancel()
            del task.embedder
            # the process will still continue in the background - it will
            # wait current waiting response to come back but then will stop

            self.auto_commit_widget.setDisabled(False)
            self.cancel_button.setDisabled(True)
            self.progressBarFinished()
            self.setBlocking(False)
            self.cb_image_attr.setDisabled(False)
            self.cb_embedder.setDisabled(False)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    from orangecontrib.imageanalytics.widgets.tests.utils import load_images
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(list(argv) if argv else [])

    data = load_images()
    widget = OWImageEmbedding()
    widget.show()
    assert QSignalSpy(widget.blockingStateChanged).wait()
    widget.set_data(data)
    widget.handleNewSignals()
    app.exec()
    widget.set_data(None)
    widget.handleNewSignals()
    widget.saveSettings()
    widget.onDeleteWidget()
    return 0


if __name__ == '__main__':
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWImageEmbedding).run(
        Table("https://datasets.biolab.si/core/bone-healing.xlsx"))
