import torch
import numpy as np
from typing import Dict, List, Optional

from models import GenerationState, PositionData


class TokenSelector:
    """Handles token ranking based on probabilities."""

    def __init__(self, mask_id: int):
        self.mask_id = mask_id

    def apply_ranking_strategy(self, probs: torch.Tensor, strategy: str) -> torch.Tensor:
        """Apply ranking strategy to get confidence scores"""
        match strategy:
            case "low_confidence":
                return self._ranking_low_confidence(probs)
            case "high_confidence":
                return self._ranking_high_confidence(probs)
            case "low_entropy":
                return self._ranking_low_entropy(probs)
            case "high_entropy":
                return self._ranking_high_entropy(probs)
            case "random":
                return self._ranking_random(probs)
            case _:
                # Default to low confidence
                return self._ranking_low_confidence(probs)

    def _ranking_low_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """Remask tokens with low confidence (low probability)"""
        return probs.max(dim=-1)

    def _ranking_high_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """Remask tokens with high confidence (high probability) - inverted"""
        return 1.0 - probs.max(dim=-1)

    def _ranking_low_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Remask positions with low entropy (more certain distributions)"""
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def _ranking_high_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Remask positions with high entropy (less certain distributions)"""
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return -entropy

    def _ranking_random(self, probs: torch.Tensor) -> torch.Tensor:
        """Random ranking strategy"""
        probs = torch.rand(probs.shape[0], dtype=torch.float)
        return probs

    def select_tokens(self, probs, max_tokens=1, strategy="constant", threshold=None):
        match strategy:
            case "constant":
                return torch.topk(probs, max_tokens).indices
            case "threshold":
                return torch.nonzero(probs >= threshold).squeeze()[:max_tokens]
            case "factor":
                es = [threshold / (n + 1) for n in range(len(probs))]
                idx = torch.argsort(probs, descending=True)
                for i in range(len(probs)):
                    j = idx[i]
                    if i >= max_tokens - 1 or probs[j] < es[i]:
                        break
                return idx[: i + 1]
            case _:
                raise ValueError("Unknown token selection strategy")


class TokenTracker:
    """Manages generation state and coordinates token selection."""

    def __init__(self, device: str, mask_id: int):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mask_id = mask_id
        self.state_cache = {}
        self.current_state = None
        self.token_selector = TokenSelector(mask_id)

    def initialize_generation(
        self, prompt_tokens: List[int], gen_length: int = 128, block_length: int = 32, tokenizer=None
    ) -> GenerationState:
        """Initialize generation state"""
        if tokenizer is None:
            raise ValueError("Tokenizer is required for generation initialization")

        x = np.full(len(prompt_tokens) + gen_length, self.mask_id, dtype=np.int)
        x[: len(prompt_tokens)] = prompt_tokens

        positions = []
        for i in range(len(x)):
            is_masked = i >= len(prompt_tokens)
            is_block_boundary = (i >= len(prompt_tokens)) and ((i - len(prompt_tokens)) % block_length == 0)

            positions.append(
                PositionData(
                    is_prompt=not is_masked,
                    is_masked=is_masked,
                    is_block_boundary=is_block_boundary,
                    candidates=[],
                )
            )

        self.current_state = GenerationState(
            step=0,
            token_ids=x.tolist(),
            tokens=[tokenizer.decode([t], skip_special_tokens=False) for t in x],
            positions=positions,
            prompt_length=len(prompt_tokens),
            block_start=len(prompt_tokens),
            block_length=block_length,
            selected_positions={},
        )

        self.state_cache[0] = self.current_state
        return self.current_state

    def apply_selections(
        self,
        selections: Dict[int, int],
        tokenizer=None,
    ) -> GenerationState:
        """Step the generation process"""
        if not self.current_state:
            raise ValueError("No generation initialized")

        if not selections:
            raise ValueError("Must provide selections")

        # Apply the provided selections to current state
        self.current_state.selected_positions.update(selections)

        tokens = self.current_state.token_ids.copy()
        for pos, tid in selections.items():
            if 0 <= pos < len(tokens):
                tokens[pos] = tid

        # Update state
        self.current_state.step += 1
        self.current_state.token_ids = tokens.tolist()
        if tokenizer:
            self.current_state.tokens = [tokenizer.decode([t], skip_special_tokens=False) for t in tokens]
        else:
            # If no tokenizer, keep existing tokens or use string representation
            self.current_state.tokens = [str(t) for t in tokens]

        # Cache state
        self.state_cache[self.current_state.step] = self.current_state

        return self.current_state

    def auto_select(
        self,
        probabilities,
        x0,
        strategy="low_confidence",
        max_tokens=1,
        selection_method="constant",
        selection_param=None,
        block_length=None,
    ):
        token_ids = self.current_state.token_ids
        if len(probabilities) != len(token_ids) or len(x0) != len(token_ids):
            raise ValueError("Probabilities and x0 should be the same length as tokens")

        if block_length is not None:
            self.current_state.block_length = block_length

        start = self.current_state.block_start
        end = start + self.current_state.block_length
        tokens = torch.tensor(token_ids).to(self.device)
        if torch.all(tokens[start:end] != self.mask_id):
            start = end
            end = start + self.current_state.block_length

        prob_window = self.token_selector.apply_ranking_strategy(probabilities[start:end], strategy)
        selected = self.token_selector.select_tokens(prob_window, max_tokens=max_tokens, strategy=selection_method, threshold=selection_param)
        candidates = prob_window.argmax(dim=-1)
        
        selected_tokens = {i+start: x0[i+start, candidates[i]] for i in selected}
        return selected_tokens


    def rewind_to_step(self, step: int) -> Optional[GenerationState]:
        """Rewind generation to a previous step"""
        if step in self.state_cache:
            self.current_state = self.state_cache[step]
            return self.current_state
        return None
