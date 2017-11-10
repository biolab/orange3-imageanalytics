import logging
import os.path
import traceback
from types import SimpleNamespace as namespace

import numpy as np
from AnyQt.QtCore import Qt, QTimer, QThread, QThreadPool
from AnyQt.QtCore import pyqtSlot as Slot
from AnyQt.QtTest import QSignalSpy
from AnyQt.QtWidgets import QLayout, QPushButton, QStyle

from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.gui import hBox
from Orange.widgets.gui import widgetBox, widgetLabel, comboBox, auto_commit
from Orange.widgets.settings import Setting
from Orange.widgets.utils import concurrent as qconcurrent
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget, Default

from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
from orangecontrib.imageanalytics.image_embedder import MODELS as EMBEDDERS_INFO


class _Input:
    IMAGES = 'Images'


class _Output:
    EMBEDDINGS = 'Embeddings'
    SKIPPED_IMAGES = 'Skipped Images'


class OWImageEmbedding(OWWidget):
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
        self._image_embedder = None
        self._executor = qconcurrent.ThreadExecutor(
            self, threadPool=QThreadPool(maxThreadCount=1)
        )
        self.setBlocking(True)
        QTimer.singleShot(0, self._init_server_connection)

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

        self.cb_embedder = comboBox(
            widget=widget_box,
            master=self,
            value='cb_embedder_current_id',
            label='Embedder:',
            orientation=Qt.Horizontal,
            callback=self._cb_embedder_changed
        )
        self.cb_embedder.setModel(VariableListModel(
            [EMBEDDERS_INFO[e]['name'] for e in self.embedders]))
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

    def _init_server_connection(self):
        self.setBlocking(False)
        self._image_embedder = ImageEmbedder(
            model=self.embedders[self.cb_embedder_current_id],
            layer='penultimate'
        )
        self._set_server_info(
            self._image_embedder.is_connected_to_server()
        )

    def set_data(self, data):
        if not data:
            self._input_data = None
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            self.input_data_info.setText(self._NO_DATA_INFO_TEXT)
            return

        self._image_attributes = ImageEmbedder.filter_image_attributes(data)
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
        self.input_data_info.setText(
            "Data with {:d} instances.".format(len(data)))

        self._cb_image_attr_changed()

    def _cb_image_attr_changed(self):
        self.commit()

    def _cb_embedder_changed(self):
        current_embedder = self.embedders[self.cb_embedder_current_id]
        self._image_embedder = ImageEmbedder(
            model=current_embedder,
            layer='penultimate'
        )
        self.embedder_info.setText(
            EMBEDDERS_INFO[current_embedder]['description'])
        if self._input_data:
            self.input_data_info.setText(
                "Data with {:d} instances.".format(len(self._input_data)))
            self.commit()
        else:
            self.input_data_info.setText(self._NO_DATA_INFO_TEXT)

    def commit(self):
        if self._task is not None:
            self.cancel()

        if self._image_embedder is None:
            self._set_server_info(connected=False)
            return

        if not self._image_attributes or self._input_data is None:
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            return

        self._set_server_info(connected=True)
        self.cancel_button.setDisabled(False)
        self.cb_image_attr.setDisabled(True)
        self.cb_embedder.setDisabled(True)

        file_paths_attr = self._image_attributes[self.cb_image_attr_current_id]
        file_paths = self._input_data[:, file_paths_attr].metas.flatten()
        origin = file_paths_attr.attributes.get("origin", "")

        assert file_paths_attr.is_string
        assert file_paths.dtype == np.dtype('O')

        file_paths_mask = file_paths == file_paths_attr.Unknown
        file_paths_valid = file_paths[~file_paths_mask]
        file_paths_valid = np.vectorize(os.path.join)(origin, file_paths_valid)

        ticks = iter(np.linspace(0.0, 100.0, file_paths_valid.size))
        set_progress = qconcurrent.methodinvoke(
            self, "__progress_set", (float,))

        def advance(success=True):
            if success:
                set_progress(next(ticks))

        def cancel():
            task.future.cancel()
            task.cancelled = True
            task.embedder.cancelled = True

        embedder = self._image_embedder

        def run_embedding(paths):
            return embedder(
                file_paths=paths, image_processed_callback=advance)

        self.auto_commit_widget.setDisabled(True)
        self.progressBarInit(processEvents=None)
        self.progressBarSet(0.0, processEvents=None)
        self.setBlocking(True)

        f = self._executor.submit(run_embedding, file_paths_valid)
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
        self.progressBarFinished(processEvents=None)
        self.setBlocking(False)

        try:
            embeddings = f.result()
        except ConnectionError:
            self._log.exception("Error", exc_info=True)
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            self._set_server_info(connected=False)
            return
        except Exception as err:
            self._log.exception("Error", exc_info=True)
            self.error("\n".join(traceback.format_exception_only(type(err), err)))
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            return

        assert self._input_data is not None
        assert len(self._input_data) == len(task.file_paths_mask)

        # Missing paths/urls were filtered out. Restore the full embeddings
        # array from information stored in task.file_path_mask ...
        embeddings_all = [None] * len(task.file_paths_mask)
        for i, embedding in zip(np.flatnonzero(~task.file_paths_mask),
                                embeddings):
            embeddings_all[i] = embedding
        embeddings_all = np.array(embeddings_all)
        self._send_output_signals(embeddings_all)

    def _send_output_signals(self, embeddings):
        embedded_images, skipped_images, num_skipped =\
            ImageEmbedder.prepare_output_data(self._input_data, embeddings)
        self.send(_Output.SKIPPED_IMAGES, skipped_images)
        self.send(_Output.EMBEDDINGS, embedded_images)
        if num_skipped is not 0:
            self.input_data_info.setText(
                "Data with {:d} instances, {:d} images skipped.".format(
                    len(self._input_data), num_skipped))

    def _set_server_info(self, connected):
        self.clear_messages()
        if connected:
            self.connection_info.setText("Connected to server.")
        else:
            self.connection_info.setText("No connection with server.")
            self.warning("Click Apply to try again.")

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()
        if self._image_embedder is not None:
            self._image_embedder.__exit__(None, None, None)

    def cancel(self):
        if self._task is not None:
            task, self._task = self._task, None
            task.cancel()
            # wait until done
            try:
                task.future.exception()
            except qconcurrent.CancelledError:
                pass

            self.auto_commit_widget.setDisabled(False)
            self.cancel_button.setDisabled(True)
            self.progressBarFinished(processEvents=None)
            self.setBlocking(False)
            self.cb_image_attr.setDisabled(False)
            self.cb_embedder.setDisabled(False)
            self._image_embedder.cancelled = False
            # reset the connection.
            connected = self._image_embedder.reconnect_to_server()
            self._set_server_info(connected=connected)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "zoo-with-images"
    data = Table(filename)
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
    import sys
    sys.exit(main(sys.argv))
