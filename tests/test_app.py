import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_empty_url_shows_validation_error(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

        app.button[0].click().run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(app.error[0].value, "Enter a URL to analyze.")

    def test_url_analysis_completes_without_streamlit_exception(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("example.com/login")

        app.button[0].click().run(timeout=10)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                element.value == "Likely a legitimate URL"
                for element in app.subheader
            )
        )

    def test_unsupported_scheme_shows_validation_error(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("ftp://example.com/file")

        app.button[0].click().run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(app.error[0].value, "Enter a valid HTTP or HTTPS URL.")

    def test_phishing_prediction_displays_result_without_exception(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value(
            "http://192.0.2.10/login/verify?account=9381"
        )

        app.button[0].click().run(timeout=10)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(element.value == "Likely a phishing URL" for element in app.subheader)
        )


if __name__ == "__main__":
    unittest.main()
