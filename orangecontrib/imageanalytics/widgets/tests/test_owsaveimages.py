import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import os

import scipy.sparse as sp
from AnyQt.QtWidgets import QFileDialog
from PIL import Image

from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.imageanalytics.widgets.owsaveimages import OWSaveImages, \
    SUPPORTED_FILE_FORMATS


def _raise_error(*args):
    raise IOError


class TestOWSaveImages(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWSaveImages)
        self.data = Table("https://datasets.biolab.si//core/bone-healing.xlsx")
        # reduce the dataset - take data of both classes
        self.data = self.data[16:22]

        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Close the file, the directory will be removed after the test
        self.widget.cancel()
        self.widgets.remove(self.widget)
        self.wait_until_finished()

    def test_dataset(self):
        self.widget.auto_save = True
        insum = self.widget.info.set_input_summary = Mock()
        savefile = self.widget.save_file = Mock()

        datasig = self.widget.Inputs.data
        self.send_signal(datasig, self.data)
        self.wait_until_finished()
        self.assertEqual(insum.call_args[0][0], "6")
        insum.reset_mock()
        savefile.reset_mock()

        self.widget.dirname = os.path.join(self.test_dir, "foo")
        self.widget.auto_save = False
        self.send_signal(datasig, self.data)
        self.assertEqual(insum.call_args[0][0], "6")
        savefile.assert_not_called()

        self.widget.auto_save = True
        self.send_signal(datasig, self.data)
        self.wait_until_finished()
        self.assertEqual(insum.call_args[0][0], "6")
        savefile.assert_called()

        self.send_signal(datasig, None)
        insum.assert_called_with(self.widget.info.NoInput)

    def test_initial_start_dir(self):
        self.widget.dirname = os.path.join(self.test_dir, "foo")
        self.assertEqual(self.widget._initial_start_dir(),
                         self.test_dir)

        with patch("os.path.exists", return_value=True):
            self.widget.dirname = os.path.join(self.test_dir, "foo")
            self.assertEqual(self.widget._initial_start_dir(), self.test_dir)

            self.widget.dirname = ""
            self.widget.last_dir = os.path.join(self.test_dir, "bar")
            self.assertEqual(
                self.widget._initial_start_dir(),
                os.path.join(self.test_dir, "bar"))

            self.widget.last_dir = os.path.join(self.test_dir, "bar")
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertEqual(self.widget._initial_start_dir(),
                             os.path.join(self.test_dir, "bar"))

            self.widget.last_dir = ""
            self.assertEqual(self.widget._initial_start_dir(),
                             os.path.expanduser(os.path.join("~", "")))

    @patch("orangecontrib.imageanalytics.widgets.owsaveimages."
           "QFileDialog.getSaveFileName")
    def test_save_file_sets_name(self, _filedialog):
        widget = self.widget

        widget.dirname = os.path.join(self.test_dir, "foo", "bar")
        widget.last_dir = os.path.join(self.test_dir, "foo")

        widget._update_messages = Mock()
        widget.save_file = Mock()

        widget.get_save_filename = Mock(return_value="")
        widget.save_file_as()
        self.assertEqual(
            widget.dirname, os.path.join(self.test_dir, "foo", "bar"))
        self.assertEqual(widget.last_dir, os.path.join(self.test_dir, "foo"))
        widget._update_messages.assert_not_called()
        widget.save_file.assert_not_called()

        bar_bar = os.path.join(self.test_dir, "bar", "bar")

        widget.get_save_filename = Mock(return_value=bar_bar)
        widget.save_file_as()
        self.assertEqual(widget.dirname, bar_bar)
        self.assertEqual(widget.last_dir, os.path.dirname(bar_bar))
        self.assertIn("bar", widget.bt_save.text())
        widget._update_messages.assert_called()
        widget.save_file.assert_called()

        widget.get_save_filename = Mock(return_value="")
        widget.save_file_as()
        self.assertEqual(widget.dirname, bar_bar)
        self.assertEqual(widget.last_dir, os.path.dirname(bar_bar))
        self.assertIn("bar", widget.bt_save.text())
        widget._update_messages.assert_called()
        widget.save_file.assert_called()

    def test_save_file_calls_save_as(self):
        widget = self.widget
        widget.save_file_as = Mock()

        self.send_signal(widget.Inputs.data, self.data)

        widget.dirname = ""
        widget.save_file()
        widget.save_file_as.assert_called()
        widget.save_file_as.reset_mock()

        widget.dirname = "bar"
        widget.save_file()
        widget.save_file_as.assert_not_called()

    def test_save_file_checks_can_save(self):
        widget = self.widget
        widget.get_save_filename = Mock(return_value="")
        widget.save_images = Mock()

        widget.save_file()
        self.wait_until_finished()
        widget.save_images.assert_not_called()

        widget.dirname = "foo"
        widget.save_file()
        self.wait_until_finished()
        # data is still none
        widget.save_images.assert_not_called()

        widget.dirname = ""
        self.send_signal(widget.Inputs.data, self.data)
        self.wait_until_finished()
        widget.save_file()
        # name empty
        widget.save_images.assert_not_called()

        widget.dirname = "foo"
        widget.save_file()
        self.wait_until_finished()
        widget.save_images.assert_called()
        widget.save_images.reset_mock()

        self.data.X = sp.csr_matrix(self.data.X)
        widget.save_file()
        widget.save_images.assert_called()

    @patch("orangecontrib.imageanalytics.widgets.owsaveimages."
           "_prepare_dir_and_save_images", Mock(side_effect=_raise_error))
    def test_save_file_write_errors(self):
        widget = self.widget
        datasig = widget.Inputs.data

        widget.auto_save = True
        widget.dirname = os.path.join(self.test_dir, "bar", "foo")

        self.send_signal(datasig, self.data)
        self.wait_until_finished()
        self.assertTrue(widget.Error.general_error.is_shown())

    def test_file_name_label(self):
        self.widget.dirname = ""
        self.widget._update_messages()
        self.assertFalse(self.widget.Error.no_file_name.is_shown())

        self.widget.auto_save = True
        self.widget._update_messages()
        self.assertTrue(self.widget.Error.no_file_name.is_shown())

        self.widget.dirname = os.path.join(self.test_dir, "foo", "bar", "baz")
        self.widget._update_messages()
        self.assertFalse(self.widget.Error.no_file_name.is_shown())

    def test_auto_save(self):
        dirname = os.path.join(self.test_dir, "foo")
        # remove the dir if already exist
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)

        self.widget.dirname = dirname
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished()
        self.assertFalse(os.path.isdir(dirname))

        self.widget.auto_save = True
        self.widget.dirname = dirname
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished()
        self.assertTrue(os.path.isdir(dirname))

        image_path = os.path.join(dirname, "Day7", "D7-Series037_z06.png")
        self.assertTrue(os.path.isfile(image_path))

    def test_scale(self):
        dirname = os.path.join(self.test_dir, "foo")
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)

        self.widget.dirname = dirname
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished()

        # nothing should change since no auto-save
        self.assertFalse(os.path.isdir(dirname))

        self.widget.auto_save = True
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished()

        # test image size, since size not set the original image size is used
        image_path = os.path.join(dirname, "Day7", "D7-Series037_z06.png")

        with Image.open(image_path) as im:
            size = im.size
        self.assertTupleEqual((512, 512), size)

        # enable scale
        self.widget.controls.use_scale.setChecked(True)
        self.wait_until_finished()
        with Image.open(image_path) as im:
            size = im.size

        # one of the scale is used, non of the preset scale is (512, 512)
        self.assertFalse((512, 512) == size)

        # change scale
        simulate.combobox_activate_index(self.widget.controls.scale_index, 1)
        self.wait_until_finished()
        with Image.open(image_path) as im:
            size = im.size
        # second option is squeezenet with scale 227x227
        self.assertTupleEqual((227, 227), size)

    def test_save_one_class_only(self):
        """
        Edge case images from one class only.
        """
        data = self.data[:10]  # first few lines belongs to one class only

        dirname = os.path.join(self.test_dir, "foo")
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)

        self.widget.dirname = dirname
        self.widget.auto_save = True
        self.send_signal(self.widget.Inputs.data, data)

    def test_save_no_class(self):
        """
        Edge case images from one class only.
        """
        data = Table(Domain(self.data.domain.attributes,
                            None,
                            self.data.domain.metas),
                     self.data.X, None, self.data.metas)

        dirname = os.path.join(self.test_dir, "foo")
        self.widget.dirname = dirname
        self.widget.auto_save = True
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        # test images not in subdirs - all images in primary dir

        self.assertTrue(os.path.isdir(self.test_dir))

        self.assertEqual(len(data), len(os.listdir(dirname)))

    def test_format(self):
        """
        Test all file format available
        """
        dirname = os.path.join(self.test_dir, "foo")

        self.widget.dirname = dirname
        self.widget.auto_save = True
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished()

        for i, f in enumerate(SUPPORTED_FILE_FORMATS):
            simulate.combobox_activate_index(
                self.widget.controls.file_format_index, i)
            self.wait_until_finished()

            # test image has right ending
            image_path = os.path.join(
                dirname, "Day7",
                "D7-0503-12-2-bone-inj-d7-3-0020-m1.{}".format(f))
            self.assertTrue(os.path.isfile(image_path))

    def test_minimum_size(self):
        pass


@unittest.skipUnless(sys.platform in ("darwin", "win32"),
                     "Test for native dialog on Windows and macOS")
class TestOWSaveImagesDarwinDialog(TestOWSaveImages):

    @patch("orangecontrib.imageanalytics.widgets.owsaveimages.QFileDialog")
    def test_get_save_filename_darwin(self, dlg):
        widget = self.widget
        widget._initial_start_dir = lambda: "baz"
        instance = dlg.return_value
        instance.exec.return_value = dlg.Accepted = QFileDialog.Accepted
        instance.selectedFiles.return_value = ["foo"]
        self.assertEqual(widget.get_save_filename(), "foo")
        self.assertEqual(dlg.call_args[0][2], "baz")

        instance.exec.return_value = dlg.Rejected = QFileDialog.Rejected
        self.assertEqual(widget.get_save_filename(), None)

    @patch("orangecontrib.imageanalytics.widgets.owsaveimages.QFileDialog")
    @patch("os.path.exists", new=lambda x: x == "old")
    @patch("orangecontrib.imageanalytics.widgets.owsaveimages.QMessageBox")
    def test_save_file_dialog_asks_for_overwrite_darwin(self, msgbox, dlg):
        def selected_files():
            nonlocal attempts
            attempts += 1
            return [["old", "new"][attempts]]

        widget = self.widget
        widget._initial_start_dir = lambda: "baz"

        instance = dlg.return_value
        instance.exec.return_value = QFileDialog.Accepted
        instance.selectedFiles = selected_files

        attempts = -1
        msgbox.question.return_value = msgbox.Yes = 1
        self.assertEqual(widget.get_save_filename(), "old")

        attempts = -1
        msgbox.question.return_value = msgbox.No = 0
        self.assertEqual(widget.get_save_filename(), "new")


if __name__ == "__main__":
    unittest.main()
