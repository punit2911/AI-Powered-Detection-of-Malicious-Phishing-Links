import unittest
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from streamlit.testing.v1 import AppTest

from app import MODEL_PATH, SHOWCASE_METRICS, load_model

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_dashboard_renders_model_analytics_charts(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 2)

    def test_showcase_metrics_are_displayed_without_classifier_comparison_chart(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        for label, value in SHOWCASE_METRICS.items():
            self.assertEqual(metrics[label], value)
        self.assertEqual(metrics["Accuracy"], "81.25%")
        self.assertFalse(
            any("Reported accuracy by classifier" in item.value for item in app.markdown)
        )

    def test_checked_in_pipeline_uses_random_forest(self):
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

        self.assertIsInstance(model.named_steps["classifier"], RandomForestClassifier)

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
                "Likely legitimate" in element.value
                for element in app.markdown
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
            any("Likely phishing" in element.value for element in app.markdown)
        )


if __name__ == "__main__":
    unittest.main()
