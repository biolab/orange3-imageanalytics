import os
import tempfile
import platform
import unittest
from unittest.mock import patch

from orangecontrib.imageanalytics.utils import atomic_update


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
