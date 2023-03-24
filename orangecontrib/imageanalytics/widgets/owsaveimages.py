import os.path
import shutil
from collections import deque
from os.path import join, isdir
from types import SimpleNamespace as namespace

from AnyQt.QtWidgets import QFileDialog, QGridLayout, QMessageBox
from AnyQt.QtCore import Qt, QSize

from Orange.data.table import Table
from Orange.widgets import gui, widget
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, OWWidget

from orangecontrib.imageanalytics.image_embedder import MODELS
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader
from orangecontrib.imageanalytics.utils.image_utils import (
    extract_paths,
    filter_image_attributes,
)

SUPPORTED_FILE_FORMATS = ["png", "jpeg", "gif", "tiff", "pdf", "bmp", "eps",
                          "ico"]


class Result(namespace):
    paths = None


def _get_classes(data):
    """
    Function retrieve classes for each image or return None if no classes.
    """
    # get classes
    return list(map(data.domain.class_var.repr_val, data.Y)) \
        if data.domain.class_var else None


def _clean_dir(dir_name):
    """
    Function removes the directory if it already exist.
    """
    if isdir(dir_name):
        shutil.rmtree(dir_name)


def _create_dir(path):
    """
    Function checks if dir exist and creates it if it is required.
    """
    dir_path = os.path.dirname(path)
    if not isdir(dir_path):
        os.makedirs(dir_path)


def _save_an_image(loader, origin, save_path, target_size):
    """
    This function loads and saves a separate image.
    """
    _create_dir(save_path)
    image = loader.load_image_or_none(origin, target_size)
    if image is not None:
        image.save(save_path)


def _prepare_dir_and_save_images(
        paths_queue, dir_name, target_size, previously_saved, state: TaskState):
    """
    This function prepares a directory structure and calls function
    that saves images.

    Parameters
    ----------
    previously_saved : int
        Number of saved images in the previous process. If the process is
        resumed it is non-zero.
    """
    res = Result(paths=paths_queue)
    if previously_saved == 0:
        _clean_dir(dir_name)

    steps = len(paths_queue) + previously_saved
    loader = ImageLoader()
    while res.paths:
        from_path, to_path = res.paths.popleft()
        _save_an_image(loader, from_path, to_path, target_size)

        state.set_progress_value((1 - len(res.paths) / steps) * 100)
        state.set_partial_result(res)
        if state.is_interruption_requested():
            return res

    return res


class OWSaveImages(OWWidget, ConcurrentWidgetMixin):
    name = "Save Images"
    description = "Save images in the directory structure."
    icon = "icons/SaveImages.svg"
    keywords = "save images, save, images"

    userhome = os.path.expanduser(f"~{os.sep}")

    class Inputs:
        data = Input("Data", Table)

    class Error(widget.OWWidget.Error):
        no_file_name = widget.Msg("Directory name is not set.")
        general_error = widget.Msg("{}")
        no_image_attribute = widget.Msg(
            "Data does not have any image attribute.")

    want_main_area = False
    resizing_enabled = False

    last_dir: str = Setting("")
    dirname: str = Setting("", schema_only=True)
    auto_save: bool = Setting(False)
    use_scale: bool = Setting(False)
    scale_index: int = Setting(0)
    file_format_index: int = Setting(0)
    image_attr_current_index: int = Setting(0)

    image_attributes = None
    data = None
    paths_queue = None
    # number of images saved in the previous iteration if process was
    # interrupted
    previously_saved = 0

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.available_scales = sorted(
            MODELS.values(), key=lambda x: x["order"])

        # create grid
        grid = QGridLayout()
        gui.widgetBox(self.controlArea, orientation=grid)

        # image attribute selection
        hbox_attr = gui.hBox(None)
        self.cb_image_attr = gui.comboBox(
            widget=hbox_attr,
            master=self,
            value='image_attr_current_index',
            label='Image attribute',
            orientation=Qt.Horizontal,
            callback=self.setting_changed
        )
        grid.addWidget(hbox_attr, 0, 0, 1, 2)

        # Scale images option
        hbox_scale = gui.hBox(None)
        gui.checkBox(
            widget=hbox_scale,
            master=self,
            value="use_scale",
            label="Scale images to ",
            callback=self.setting_changed
        )
        self.scale_combo = gui.comboBox(
            widget=hbox_scale,
            master=self,
            value="scale_index",
            items=["{} ({}×{})".format(v["name"], *v["target_image_size"])
                   for v in self.available_scales],
            callback=self.setting_changed
        )
        grid.addWidget(hbox_scale, 1, 0, 1, 2)

        # file format
        hbox_format = gui.hBox(None)
        gui.comboBox(
            widget=hbox_format,
            master=self,
            value="file_format_index",
            label="File format",
            items=[x.upper() for x in SUPPORTED_FILE_FORMATS],
            orientation=Qt.Horizontal,
            callback=self.setting_changed
        )
        grid.addWidget(hbox_format, 2, 0, 1, 2)

        # auto save
        grid.addWidget(
            gui.checkBox(
                widget=None,
                master=self,
                value="auto_save",
                label="Autosave when receiving new data or settings change",
                callback=self._update_messages),
            3, 0, 1, 2)

        # buttons
        self.bt_save = gui.button(self.buttonsArea, self, "Save", callback=self.save_file)
        gui.button(self.buttonsArea, self, "Save as ...", callback=self.save_file_as)

        self.scale_combo.setEnabled(self.use_scale)
        self.adjustSize()
        self._update_messages()

    def setting_changed(self):
        """
        When any setting changes save files if auto_save.
        """
        self.scale_combo.setEnabled(self.use_scale)
        self.reset_queue()
        if self.auto_save:
            self.save_file()

    def gather_paths(self):
        classes = _get_classes(self.data)
        file_paths_attr = self.image_attributes[self.image_attr_current_index]
        file_paths = extract_paths(self.data, file_paths_attr)
        file_format = SUPPORTED_FILE_FORMATS[self.file_format_index]
        from_paths, to_paths = [], []

        for i, path in enumerate(file_paths):
            from_paths.append(path)

            filename = os.path.basename(path)
            if "." in filename:
                filename = filename.rsplit(".")[0]  # remove file ending
            to_paths.append(join(join(self.dirname, classes[i])
                                 if classes is not None else self.dirname,
                                 "{}.{}".format(filename, file_format)))
        return from_paths, to_paths

    def save_images(self):
        if self.paths_queue is None:
            from_path, to_path = self.gather_paths()
            self.paths_queue = deque(list(zip(from_path, to_path)))
        image_size = \
            self.available_scales[self.scale_index]["target_image_size"]\
            if self.use_scale else None
        self.bt_save.setText("Stop")
        self.start(_prepare_dir_and_save_images, self.paths_queue,
                   self.dirname, image_size, self.previously_saved)

    def on_partial_result(self, result: Result):
        self.paths_queue = result.paths
        self.previously_saved += 1

    def on_done(self, result: Result):
        assert len(result.paths) == 0
        self.bt_save.setText("Save")
        self.reset_queue()

    def on_exception(self, ex: Exception):
        self.Error.general_error(ex)
        self.bt_save.setText("Save")
        self.reset_queue()

    def reset_queue(self):
        self.paths_queue = None
        self.previously_saved = 0

    def save_file(self):
        """
        This function is called when save button is pressed.
        """
        # if task already running interrupt it
        if self.task is not None:
            self.cancel()
            self.bt_save.setText("Resume save")
        # when task not running start/restart it
        else:
            if not self.dirname:
                self.save_file_as()
                return

            self.Error.general_error.clear()
            if self.data is None or self.image_attributes is None \
                    or not self.dirname:
                return
            self.save_images()

    def save_file_as(self):
        """
        It is called when save as button is pressed or save button is pressed
        and path not set yet.
        """
        dirname = self.get_save_filename()
        if not dirname:
            return
        self.dirname = dirname
        self.last_dir = os.path.split(self.dirname)[0]
        self.bt_save.setText(f"Save as {os.path.basename(dirname)}")
        self._update_messages()
        self.reset_queue()
        self.save_file()

    @Inputs.data
    def dataset(self, data):
        """
        When new data receives update statuses and error messages. If aut_save
        also save them.
        """
        self.Error.clear()
        self.data = data
        self._update_status()
        self._update_messages()
        self.image_attributes = (
            filter_image_attributes(data) if data is not None else []
        )
        self._update_image_attributes()
        if self.auto_save and self.dirname:
            self.save_file()

    def _update_image_attributes(self):
        """
        Function updates attribute drop-down when new data comes.
        """
        self.cb_image_attr.setModel(VariableListModel(self.image_attributes))
        self.cb_image_attr.setCurrentIndex(self.image_attr_current_index)
        if not self.image_attributes:
            self.Error.no_image_attribute()

    def _update_messages(self):
        """
        Updates messages.
        """
        self.Error.no_file_name(
            shown=not self.dirname and self.auto_save)

    def _update_status(self):
        """
        Update input summary.
        """
        if self.data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(
                str(len(self.data)),
                f"Data set {self.data.name or '(no name)'} "
                f"with {len(self.data)} images.")

    def _initial_start_dir(self):
        """
        Function returns the initial directory for a dialog.
        """
        if self.dirname and os.path.exists(os.path.split(self.dirname)[0]):
            return os.path.join(os.path.dirname(self.dirname))
        else:
            return os.path.join(self.last_dir or self.userhome)

    def get_save_filename(self):
        """
        Open a user dialog and returns the dicrectory path.
        """
        filename = self._initial_start_dir()
        while True:
            dlg = QFileDialog(
                None, "Select directory to save", filename)
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly)
            dlg.setAcceptMode(dlg.AcceptSave)
            dlg.setOption(QFileDialog.DontConfirmOverwrite)
            if dlg.exec() == QFileDialog.Rejected:
                return None
            filename = list(dlg.selectedFiles())[0]
            if not os.path.exists(filename) or QMessageBox.question(
                    self, "Overwrite file?",
                    f"Folder {os.path.split(filename)[1]} already exists.\n"
                    "Overwrite?") == QMessageBox.Yes:
                return filename

    def clear(self):
        super().clear()
        self.cancel()

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def sizeHint(self):
        return QSize(500, 450)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSaveImages).run(
        Table("https://datasets.biolab.si/core/bone-healing.xlsx"))
