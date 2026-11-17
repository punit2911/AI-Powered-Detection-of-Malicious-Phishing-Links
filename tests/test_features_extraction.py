import unittest
from unittest.mock import patch

from features_extraction import MODEL_FEATURE_NAMES, Extractor


class ModelFeatureExtractionTests(unittest.TestCase):
    def setUp(self):
        self.extractor = Extractor()

    def test_features_match_training_schema_without_network_access(self):
        url = "https://login.example.com/account/reset-password?id=7&source=mail"

        with patch("features_extraction.requests.get") as request:
            features = self.extractor.extract_model_features(url)

        self.assertEqual(tuple(features), MODEL_FEATURE_NAMES)
        self.assertEqual(features["PathLength"], len("/account/reset-password"))
        self.assertEqual(features["UrlLength"], len(url))
        self.assertEqual(features["HostnameLength"], len("login.example.com"))
        self.assertEqual(features["NumDots"], 2)
        self.assertEqual(features["SubdomainLevel"], 3)
        self.assertEqual(features["IpAddress"], 0)
        request.assert_not_called()

    def test_detects_ipv4_hostname(self):
        features = self.extractor.extract_model_features(
            "http://192.0.2.10/login"
        )

        self.assertEqual(features["IpAddress"], 1)


if __name__ == "__main__":
    unittest.main()
