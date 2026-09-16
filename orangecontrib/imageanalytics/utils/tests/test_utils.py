import os
import tempfile
import platform
import unittest
from unittest.mock import patch, MagicMock

import requests

from orangecontrib.imageanalytics.utils import atomic_update, download_url_to_file


class TestAtomicUpdate(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.target_file = os.path.join(self.temp_dir, "test_file.txt")

    def tearDown(self):
        # Clean up temporary directory and files
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_atomic_update_basic(self):
        """Test basic atomic update functionality."""
        content = b"hello world"
        with atomic_update(self.target_file, mode="wb") as f:
            f.write(content)

        # Verify the target file contains the expected content
        with open(self.target_file, "rb") as f:
            self.assertEqual(f.read(), content)

    @unittest.skipUnless(platform.system() != "Windows", "Test skipped on Windows")
    def test_atomic_update_permissions(self):
        """Test that permissions are preserved during atomic update."""
        content = b"hello world"
        # Ensure target_file exists
        with open(self.target_file, "wb") as f:
            pass
        # Set a specific mode for the target file
        os.chmod(self.target_file,0o600)

        with atomic_update(self.target_file, mode="wb") as f:
            f.write(content)

        # Verify the permissions are preserved
        self.assertEqual(os.stat(self.target_file).st_mode & 0o777, 0o600)

    def test_atomic_update_exception_handling(self):
        """Test that the temporary file is deleted if an exception occurs."""
        content = b"hello world"

        # Mock os.mkstemp to return a specific file path
        with patch("orangecontrib.imageanalytics.utils.tempfile.mkstemp") as mock_mkstemp:
            mock_mkstemp.return_value = (123, os.path.join(self.temp_dir, "temp_file.txt"))

            # Mock os.replace to raise an exception
            with patch("orangecontrib.imageanalytics.utils.os.replace", side_effect=OSError("Error replacing file")):
                with self.assertRaises(OSError):
                    with atomic_update(self.target_file, mode="wb") as f:
                        f.write(content)
                        raise OSError("Test exception")

                # Verify the temporary file was deleted
                self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "temp_file.txt")))

    def test_atomic_update_nonexistent_target(self):
        """Test behavior when the target file does not exist."""
        content = b"hello world"

        # Ensure the target file does not exist
        if os.path.exists(self.target_file):
            os.remove(self.target_file)

        with atomic_update(self.target_file, mode="wb") as f:
            f.write(content)

        # Verify the target file was created
        self.assertTrue(os.path.exists(self.target_file))
        with open(self.target_file, "rb") as f:
            self.assertEqual(f.read(), content)


class TestDownloadUrlToFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_file.txt")

    def tearDown(self):
        # Clean up temporary directory and files
        import shutil
        shutil.rmtree(self.temp_dir)

    def _mock_response(self):
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = iter([b"chunk1", b"chunk2"])
        mock_response.raise_for_status.return_value = None
        mock_response.__enter__ = lambda self: mock_response
        return mock_response

    def test_successful_download(self):
        """Test successful download of a file."""
        mock_response = self._mock_response()
        with patch("orangecontrib.imageanalytics.utils.requests.get", return_value=mock_response):
            download_url_to_file("http://example.com/file", self.test_file)

        # Verify the file was created and contains the expected content
        with open(self.test_file, "rb") as f:
            content = f.read()
            self.assertEqual(content, b"chunk1chunk2")

    def test_download_with_progress_callback(self):
        """Test download with progress callback."""
        mock_response = self._mock_response()

        progress_called = False
        progress_args = None

        def mock_progress_callback(count, size):
            nonlocal progress_called, progress_args
            progress_called = True
            progress_args = (count, size)

        with patch("orangecontrib.imageanalytics.utils.requests.get", return_value=mock_response):
            download_url_to_file(
                "http://example.com/file",
                self.test_file,
                progress_callback=mock_progress_callback
            )

        self.assertTrue(progress_called)
        self.assertEqual(progress_args, (12, 100))

    def test_download_with_force(self):
        """Test download with force=True."""
        mock_response = self._mock_response()

        # Create a dummy file initially
        with open(self.test_file, "w") as f:
            f.write("dummy content")

        with patch("orangecontrib.imageanalytics.utils.requests.get", return_value=mock_response):
            download_url_to_file("http://example.com/file", self.test_file, force=True)

        # Verify the file was updated
        with open(self.test_file, "rb") as f:
            content = f.read()
            self.assertEqual(content, b"chunk1chunk2")

    def test_download_without_force(self):
        """Test download with force=False (should not overwrite existing file)."""
        mock_response = self._mock_response()

        # Create a dummy file initially
        with open(self.test_file, "w") as f:
            f.write("dummy content")

        with patch("orangecontrib.imageanalytics.utils.requests.get", return_value=mock_response):
            download_url_to_file("http://example.com/file", self.test_file, force=False)

        # Verify the file was not overwritten
        with open(self.test_file, "rb") as f:
            content = f.read()
            self.assertEqual(content, b"dummy content")

    def test_download_error_handling(self):
        """Test error handling for invalid URLs or network errors."""
        mock_response = self._mock_response()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Bad Request")

        with patch("orangecontrib.imageanalytics.utils.requests.get", return_value=mock_response):
            with self.assertRaises(requests.exceptions.HTTPError):
                download_url_to_file("http://example.com/file", self.test_file)

        # Verify the file was not created
        self.assertFalse(os.path.exists(self.test_file))

    def test_download_without_content_length(self):
        """Test download without content-length header."""
        mock_response = self._mock_response()
        mock_response.headers = {}

        with patch("orangecontrib.imageanalytics.utils.requests.get", return_value=mock_response):
            download_url_to_file("http://example.com/file", self.test_file)

        # Verify the file was created and contains the expected content
        with open(self.test_file, "rb") as f:
            content = f.read()
            self.assertEqual(content, b"chunk1chunk2")
