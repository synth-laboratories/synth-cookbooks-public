"""Credential-free configuration checks; not HealthBench runtime acceptance."""

from pathlib import Path
import tomllib
import unittest


class ProviderConfigTests(unittest.TestCase):
    def test_current_policy_uses_openrouter(self):
        root = Path(__file__).parent
        config = tomllib.loads((root / "gepa.toml").read_text())
        policy = config["policy"]
        self.assertEqual(policy["provider"], "openrouter")
        self.assertEqual(policy["model"], "meta-llama/llama-3.1-8b-instruct")
        self.assertEqual(policy["api_key_env"], "OPENROUTER_API_KEY")
        self.assertEqual(policy["base_url"], "https://openrouter.ai/api/v1")
        self.assertNotIn("groq", config["cache"]["namespace"])
        self.assertIn("parked", (root / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
