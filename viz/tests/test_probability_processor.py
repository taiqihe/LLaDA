"""Unit tests for ProbabilityProcessor component."""

import pytest
import torch
from unittest.mock import Mock

from probability_processor import ProbabilityProcessor
from models import TokenCandidate


class TestProbabilityProcessor:
    """Test cases for the ProbabilityProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.processor = ProbabilityProcessor(self.device)

        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.decode.side_effect = lambda tokens, **kwargs: f"token_{tokens[0]}"

    def test_logits_to_probabilities_basic(self):
        """Test basic logits to probabilities conversion."""
        logits = torch.tensor([1.0, 2.0, 3.0])

        x0, probs = self.processor.logits_to_probabilities(logits)

        # x0 should be argmax
        assert x0.item() == 2  # Index of maximum logit

        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(), torch.tensor(1.0))

        # Should be proper probabilities
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_logits_to_probabilities_with_temperature(self):
        """Test temperature scaling."""
        logits = torch.tensor([1.0, 2.0, 3.0])

        # High temperature should flatten distribution
        _, probs_high_temp = self.processor.logits_to_probabilities(logits, softmax_temperature=2.0)

        # Low temperature should sharpen distribution
        _, probs_low_temp = self.processor.logits_to_probabilities(logits, softmax_temperature=0.5)

        # High temperature should be less concentrated
        assert probs_high_temp.max() < probs_low_temp.max()

    def test_gumbel_noise(self):
        """Test Gumbel noise application."""
        logits = torch.tensor([1.0, 2.0, 3.0])

        # Without noise
        result1 = self.processor.add_gumbel_noise(logits, temperature=0.0)
        torch.testing.assert_close(result1, logits)

        # With noise - should be different
        result2 = self.processor.add_gumbel_noise(logits, temperature=1.0)
        assert not torch.allclose(result1, result2)

    def test_apply_top_p_filtering(self):
        """Test nucleus (top-p) sampling."""
        # Create probabilities where top-p should filter some tokens
        probs = torch.tensor([0.5, 0.3, 0.15, 0.05])  # Already normalized

        # top_p = 0.8 should keep first two tokens (0.5 + 0.3 = 0.8)
        filtered = self.processor.apply_top_p_filtering(probs, top_p=0.8)

        # Should zero out low probability tokens
        assert filtered[0] > 0  # Keep
        assert filtered[1] > 0  # Keep
        assert filtered[2] == 0  # Filter
        assert filtered[3] == 0  # Filter

        # Should still sum to 1 after renormalization
        assert torch.allclose(filtered.sum(), torch.tensor(1.0))

    def test_apply_top_p_filtering_edge_cases(self):
        """Test edge cases for top-p filtering."""
        probs = torch.tensor([0.6, 0.2, 0.1, 0.1])

        # top_p = 1.0 should keep all tokens
        filtered = self.processor.apply_top_p_filtering(probs, top_p=1.0)
        torch.testing.assert_close(filtered, probs)

        # Very low top_p should keep at least one token
        filtered = self.processor.apply_top_p_filtering(probs, top_p=0.1)
        assert filtered[0] > 0  # Always keep the highest
        assert (filtered[1:] == 0).all()  # Remove others

    def test_apply_token_restrictions(self):
        """Test combined top_k and top_p restrictions."""
        # Create probabilities for testing
        probs = torch.tensor([0.4, 0.3, 0.2, 0.05, 0.03, 0.02])

        # Test case where top_k is more restrictive
        filtered, actual_k = self.processor.apply_token_restrictions(probs, top_k=3, top_p=0.95)
        assert actual_k == 3  # top_k is more restrictive

        # Test case where top_p is more restrictive
        filtered, actual_k = self.processor.apply_token_restrictions(probs, top_k=10, top_p=0.65)
        assert actual_k == 1  # top_p keeps first 1 token (0.4 < 0.65, but 0.4+0.3=0.7 > 0.65)

    def test_get_token_candidates(self):
        """Test basic token candidate extraction."""
        logits = torch.tensor([3.0, 1.0, 2.0, 0.5])
        probs = torch.softmax(logits, dim=0)
        top_k = 3

        candidates = self.processor.get_token_candidates(logits, probs, top_k, self.mock_tokenizer)

        assert len(candidates) == 3
        assert all(isinstance(c, TokenCandidate) for c in candidates)

        # Should be sorted by probability (highest first)
        assert candidates[0].prob >= candidates[1].prob >= candidates[2].prob

        # Check that ranks are assigned correctly
        assert candidates[0].rank == 0
        assert candidates[1].rank == 1
        assert candidates[2].rank == 2

        # Check tokenizer was called
        assert self.mock_tokenizer.decode.call_count == 3

    def test_get_token_candidates_with_invalid_probs(self):
        """Test handling of invalid probabilities."""
        # NaN probabilities
        logits = torch.tensor([1.0, float("nan"), 2.0])
        probs = torch.tensor([0.3, float("nan"), 0.7])

        candidates = self.processor.get_token_candidates(logits, probs, 2, self.mock_tokenizer)
        assert len(candidates) == 0

        # Inf probabilities
        probs_inf = torch.tensor([0.3, float("inf"), 0.7])
        candidates = self.processor.get_token_candidates(logits, probs_inf, 2, self.mock_tokenizer)
        assert len(candidates) == 0

    def test_get_token_candidates_with_restrictions(self):
        """Test token candidates with visual/actual restrictions."""
        logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.2])
        probs = torch.softmax(logits, dim=0)

        candidates, actual_k = self.processor.get_token_candidates_with_restrictions(
            logits,
            probs,
            visual_top_k=4,
            actual_top_k=2,
            top_p=0.9,
            tokenizer=self.mock_tokenizer,
        )

        # Should return visual_top_k candidates
        assert len(candidates) == 4

        # Should mark which are in actual restrictions
        actual_candidates = [c for c in candidates if c.is_in_actual]
        assert len(actual_candidates) == actual_k

        # First candidates should be marked as actual
        assert candidates[0].is_in_actual
        assert candidates[1].is_in_actual if actual_k > 1 else True

    def test_get_token_candidates_with_restrictions_edge_cases(self):
        """Test edge cases for restricted candidates."""
        logits = torch.tensor([2.0, 1.0])
        probs = torch.softmax(logits, dim=0)

        # Visual < actual should work
        candidates, actual_k = self.processor.get_token_candidates_with_restrictions(
            logits,
            probs,
            visual_top_k=1,
            actual_top_k=2,
            top_p=1.0,
            tokenizer=self.mock_tokenizer,
        )

        # Should return max(visual, actual) candidates
        assert len(candidates) == 2

        # Both should be marked as actual
        assert all(c.is_in_actual for c in candidates)

    def test_tokenizer_error_handling(self):
        """Test handling of tokenizer errors."""
        # Mock tokenizer that raises errors
        error_tokenizer = Mock()
        error_tokenizer.decode.side_effect = Exception("Tokenizer error")

        logits = torch.tensor([1.0, 2.0])
        probs = torch.softmax(logits, dim=0)

        candidates = self.processor.get_token_candidates(logits, probs, 2, error_tokenizer)

        # Should handle errors gracefully
        assert len(candidates) == 2
        assert all(c.token.startswith("<token_") for c in candidates)

    def test_reprocess_probabilities_with_settings(self):
        """Test probability reprocessing with settings."""
        # Create mock raw logits
        raw_logits = [
            {0: 2.0, 1: 1.0, 2: 0.5},  # Sparse logits for position 0
            {0: 1.5, 1: 2.5, 2: 0.8},  # Sparse logits for position 1
        ]

        settings = {
            "softmax_temperature": 0.8,
            "gumbel_temperature": 0.0,
            "apply_gumbel_noise": False,
            "visual_top_k": 3,
            "actual_top_k": 2,
            "top_p": 0.9,
        }

        result = self.processor.reprocess_probabilities_with_settings(raw_logits, settings, self.mock_tokenizer)

        assert "positions" in result
        assert "settings_applied" in result

        positions = result["positions"]
        assert len(positions) == 2

        # Check that settings were applied
        settings_applied = result["settings_applied"]
        assert settings_applied["softmax_temperature"] == 0.8
        assert settings_applied["visual_top_k"] == 3
        assert settings_applied["actual_top_k"] == 2

        # Check that candidates have is_in_actual property
        for position in positions:
            for candidate in position["candidates"]:
                assert "is_in_actual" in candidate

    def test_device_consistency(self):
        """Test that operations maintain device consistency."""
        # Test CPU device
        processor_cpu = ProbabilityProcessor("cpu")

        logits = torch.tensor([1.0, 2.0, 3.0])
        x0, probs = processor_cpu.logits_to_probabilities(logits)

        assert x0.device.type == "cpu"
        assert probs.device.type == "cpu"

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large logits
        large_logits = torch.tensor([100.0, 200.0, 150.0])
        x0, probs = self.processor.logits_to_probabilities(large_logits)

        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(), torch.tensor(1.0))

        # Very small probabilities
        small_probs = torch.tensor([1e-10, 1e-8, 1e-6])
        small_probs = small_probs / small_probs.sum()  # Normalize

        filtered = self.processor.apply_top_p_filtering(small_probs, top_p=0.5)
        assert torch.isfinite(filtered).all()
        assert torch.allclose(filtered.sum(), torch.tensor(1.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
