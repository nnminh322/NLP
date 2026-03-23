import unittest
from unittest.mock import patch

from g4k.utils import get_secret


class TestGetSecret(unittest.TestCase):
    @patch("g4k.utils.load_dotenv")
    @patch("g4k.utils.os.getenv")
    def test_get_secret_success(self, mock_getenv, mock_load_dotenv):
        # Set up the mock to return a value for a specific key
        mock_getenv.return_value = "mock_value"

        secret = get_secret("SECRET_KEY")

        # Assert that the value is correct
        self.assertEqual(secret, "mock_value")

        # Ensure load_dotenv is called for the secret path
        mock_load_dotenv.assert_called()

        # Ensure getenv was called with the correct key
        mock_getenv.assert_called_once_with("SECRET_KEY")

    @patch("g4k.utils.load_dotenv")
    @patch("g4k.utils.os.getenv", return_value=None)
    def test_get_secret_key_error(self, mock_getenv, mock_load_dotenv):
        # Test when the key does not exist
        with self.assertRaises(KeyError):
            get_secret("NON_EXISTENT_KEY")

        # Ensure getenv was called with the missing key
        mock_getenv.assert_called_once_with("NON_EXISTENT_KEY")


if __name__ == "__main__":
    unittest.main()
