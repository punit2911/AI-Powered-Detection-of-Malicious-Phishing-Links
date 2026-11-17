import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import RESEARCH_ACCURACIES, RESEARCH_KEY_METRICS, research_accuracy_chart

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_dashboard_renders_model_analytics_charts(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 3)

    def test_static_research_metrics_and_accuracy_chart_are_displayed(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        for label, value in RESEARCH_KEY_METRICS.items():
            self.assertEqual(metrics[label], f"{value:.2%}")

        chart = research_accuracy_chart().data[0]
        self.assertEqual(set(chart.y), set(RESEARCH_ACCURACIES))
        self.assertEqual(
            dict(zip(chart.y, chart.x)),
            RESEARCH_ACCURACIES,
        )

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
