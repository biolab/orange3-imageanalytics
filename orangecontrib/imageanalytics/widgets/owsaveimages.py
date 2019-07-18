import os.path
import shutil
from urllib.parse import urlparse, urljoin

from AnyQt.QtWidgets import QFileDialog, QGridLayout, QMessageBox
from AnyQt.QtCore import Qt

from Orange.data.table import Table
from Orange.widgets import gui, widget
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input

from orangecontrib.imageanalytics.image_embedder import MODELS, ImageEmbedder
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader

SUPPORTED_FILE_FORMATS = ["png", "jpeg", "gif", "tiff", "pdf", "bmp", "eps",
                          "ico"]


class OWSaveImages(widget.OWWidget):
    name = "Save Images"
    description = "Save images in the directory structure."
    icon = "icons/SaveImages.svg"
    keywords = ["save", "saveimages", "save images", "images"]

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

    def __init__(self):
        super().__init__()

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
        grid.addWidget(hbox_attr, 1, 0, 1, 2)

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
        grid.addWidget(hbox_scale, 3, 0, 1, 2)

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
        grid.addWidget(hbox_format, 4, 0, 1, 2)

        # auto save
        grid.addWidget(
            gui.checkBox(
                widget=None,
                master=self,
                value="auto_save",
                label="Autosave when receiving new data or settings change",
                callback=self._update_messages),
            5, 0, 1, 2)

        # buttons
        self.bt_save = gui.button(None, self, "Save", callback=self.save_file)
        grid.addWidget(self.bt_save, 7, 0)
        grid.addWidget(
            gui.button(None, self, "Save as ...", callback=self.save_file_as),
            7, 1)

        grid.setRowMinimumHeight(5, 8)
        grid.setRowMinimumHeight(6, 20)

        self.scale_combo.setEnabled(self.use_scale)
        self.adjustSize()
        self._update_messages()

    def setting_changed(self):
        """
        When any setting changes save files if auto_save.
        """
        self.scale_combo.setEnabled(self.use_scale)
        if self.auto_save:
            self.save_file()

    def save_file(self):
        """
        This function is called when save button is pressed.
        """
        if not self.dirname:
            self.save_file_as()
            return

        self.Error.general_error.clear()
        if self.data is None or self.image_attributes is None \
                or not self.dirname:
            return
        try:
            self._prepare_dir_and_save_images()
        except IOError as err_value:
            self.Error.general_error(str(err_value))

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
        self.save_file()

    def _prepare_dir_and_save_images(self):
        """
        This function prepares a directory structure and calls function
        that saves images.
        """
        # remove the dir if already exist
        if os.path.isdir(self.dirname):
            shutil.rmtree(self.dirname)

        # get_classes
        classes = self._get_classes()
        classes_unique = set(classes) if classes is not None else None

        # create dirs
        os.mkdir(self.dirname)
        if classes_unique is not None:
            for c in classes_unique:
                os.makedirs(os.path.join(self.dirname, c))

        # save images
        self._save_images()

    def _save_images(self):
        """
        Function extracts paths and classes and calls the function that loads
        and save separate images.
        """
        classes = self._get_classes()
        file_paths_attr = self.image_attributes[self.image_attr_current_index]
        file_paths = self.data[:, file_paths_attr].metas.flatten()
        origin = file_paths_attr.attributes.get("origin", "")
        file_paths_mask = file_paths == file_paths_attr.Unknown
        file_paths_valid = file_paths[~file_paths_mask]

        loader = ImageLoader()

        if urlparse(origin).scheme in ("http", "https", "ftp", "data") and \
                origin[-1] != "/":
            origin += "/"
        for i, a in enumerate(file_paths_valid):
            urlparts = urlparse(a)
            path = a
            if urlparts.scheme not in ("http", "https", "ftp", "data"):
                if urlparse(origin).scheme in ("http", "https", "ftp", "data"):
                    path = urljoin(origin, a)
                else:
                    path = os.path.join(origin, a)
            self._save_an_image(
                loader, path, os.path.join(self.dirname, classes[i])
                if classes is not None else self.dirname)

    def _save_an_image(self, loader, origin, save_path):
        """
        This function saves a separate image.
        """
        file_format = SUPPORTED_FILE_FORMATS[self.file_format_index]
        filename = os.path.basename(origin)
        if "." in filename:
            filename = ".".join(filename.split("."))

        image = loader.load_image_or_none(
            origin,
            (self.available_scales[self.scale_index]["target_image_size"]
             if self.use_scale else None))
        if image is not None:
            image.save(
                "{}.{}".format(os.path.join(save_path, filename), file_format),
                format=file_format)

    def _get_classes(self):
        """
        Function retrieve classes for each image or return None if no classes.
        """
        # get classes
        return list(map(self.data.domain.class_var.repr_val, self.data.Y)) \
            if self.data.domain.class_var else None

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
        self.image_attributes = ImageEmbedder.filter_image_attributes(data) if \
            data is not None else []
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


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSaveImages).run(
        Table("https://datasets.biolab.si//core/bone-healing.xlsx"))
