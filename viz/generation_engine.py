import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import DEFAULT_MASK_ID
from models import GenerationState, PositionData, GenerationParams
from probability_processor import ProbabilityProcessor


class TokenSelector:
    """Handles token selection strategies and remasking."""

    def __init__(self, mask_id: int = DEFAULT_MASK_ID):
        self.mask_id = mask_id

    def apply_remasking_strategy(
        self,
        x0: torch.Tensor,
        probs: torch.Tensor,
        strategy: str
    ) -> torch.Tensor:
        """Apply remasking strategy to get confidence scores"""
        if strategy == "low_confidence":
            return self._remasking_low_confidence(x0, probs)
        elif strategy == "high_confidence":
            return self._remasking_high_confidence(x0, probs)
        elif strategy == "low_entropy":
            return self._remasking_low_entropy(probs)
        elif strategy == "high_entropy":
            return self._remasking_high_entropy(probs)
        elif strategy == "random":
            return self._remasking_random(x0)
        else:
            # Default to low confidence
            return self._remasking_low_confidence(x0, probs)

    def _remasking_low_confidence(
        self,
        x0: torch.Tensor,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """Remask tokens with low confidence (low probability)"""
        return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

    def _remasking_high_confidence(
        self,
        x0: torch.Tensor,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """Remask tokens with high confidence (high probability) - inverted"""
        confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        return 1.0 - confidence

    def _remasking_low_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Remask positions with low entropy (more certain distributions)"""
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def _remasking_high_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Remask positions with high entropy (less certain distributions)"""
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return -entropy

    def _remasking_random(self, x0: torch.Tensor) -> torch.Tensor:
        """Random remasking strategy"""
        return torch.rand_like(x0, dtype=torch.float)

    def select_tokens_to_decode(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        probs: torch.Tensor,
        remasking_strategy: str,
        tokens_to_transfer: int,
        block_start: int,
        block_end: int
    ) -> torch.Tensor:
        """Select which tokens to decode based on remasking strategy"""
        # Get confidence scores using the remasking strategy
        confidence = self.apply_remasking_strategy(x0, probs, remasking_strategy)

        # Only consider masked positions
        mask_index = x == self.mask_id
        confidence = torch.where(mask_index, confidence, torch.tensor(-np.inf))

        # Restrict to current block
        confidence[:, :block_start] = -np.inf
        confidence[:, block_end:] = -np.inf

        # Select top tokens to transfer
        new_x = x.clone()
        if tokens_to_transfer > 0:
            _, select_indices = torch.topk(confidence[0], k=min(tokens_to_transfer, mask_index[0].sum().item()))
            transfer_mask = torch.zeros_like(x[0], dtype=torch.bool)
            transfer_mask[select_indices] = True
            new_x[0, transfer_mask] = x0[0, transfer_mask]

        return new_x


class GenerationEngine:
    """Manages generation state and coordinates token selection."""

    def __init__(self, device: str = "cuda", mask_id: int = DEFAULT_MASK_ID):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mask_id = mask_id
        self.state_cache = {}
        self.current_state = None
        self.token_selector = TokenSelector(mask_id)
        self.prob_processor = ProbabilityProcessor(device)

        # Cache for forward pass results
        self._cached_logits = None
        self._cached_probs = None
        self._cached_x0 = None
        self._current_block_length = None

    def initialize_generation(
        self,
        prompt_tokens: List[int],
        gen_length: int = 128,
        block_length: int = 32,
        tokenizer=None
    ) -> GenerationState:
        """Initialize generation state"""
        if tokenizer is None:
            raise ValueError("Tokenizer is required for generation initialization")

        x = torch.full((1, len(prompt_tokens) + gen_length), self.mask_id, dtype=torch.long).to(self.device)
        x[:, :len(prompt_tokens)] = torch.tensor(prompt_tokens).to(self.device)

        positions = []
        for i in range(x.shape[1]):
            is_masked = i >= len(prompt_tokens)
            is_block_boundary = (i >= len(prompt_tokens)) and ((i - len(prompt_tokens)) % block_length == 0)

            positions.append(PositionData(
                position=i,
                current_token=tokenizer.decode([x[0, i].item()]) if not is_masked else "[MASK]",
                current_token_id=x[0, i].item() if not is_masked else self.mask_id,
                is_masked=is_masked,
                candidates=[],
                is_block_boundary=is_block_boundary
            ))

        self.current_state = GenerationState(
            step=0,
            block=0,
            tokens=x[0].tolist(),
            positions=positions,
            prompt_length=len(prompt_tokens),
            selected_positions={}
        )

        self.state_cache[0] = self.current_state
        return self.current_state

    def step_generation(
        self,
        forward_pass_fn,
        tokenizer,
        params: GenerationParams
    ) -> Tuple[GenerationState, bool]:
        """Step the generation process"""
        if not self.current_state:
            raise ValueError("No generation initialized")

        x = torch.tensor([self.current_state.tokens]).to(self.device)
        prompt_length = self.current_state.prompt_length
        gen_length = len(self.current_state.tokens) - prompt_length

        # Apply manual selections if provided
        if params.manual_selections:
            for pos, token_id in params.manual_selections.items():
                if pos < len(self.current_state.tokens):
                    x[0, pos] = token_id
                    self.current_state.selected_positions[pos] = token_id

        num_blocks = gen_length // params.block_length
        current_block = self.current_state.block

        if current_block >= num_blocks:
            return self.current_state, True  # Generation complete

        # Get current block mask
        block_start = prompt_length + current_block * params.block_length
        block_end = prompt_length + (current_block + 1) * params.block_length
        block_mask_index = (x[:, block_start:block_end] == self.mask_id)

        if block_mask_index.sum() == 0:
            # Current block is complete, move to next
            self.current_state.block += 1
            self.current_state.step += 1
            return self.step_generation(forward_pass_fn, tokenizer, params)

        # Use the specified number of tokens to select
        tokens_to_transfer = min(params.tokens_to_select, block_mask_index.sum().item())

        # Step 1: Forward pass to get logits
        logits = forward_pass_fn(x)

        # Step 2: Convert logits to probabilities and get predicted tokens
        x0, probs = self.prob_processor.logits_to_probabilities(
            logits,
            softmax_temperature=1.0,
            gumbel_temperature=params.gumbel_temperature
        )

        # Cache results for potential token-only selection
        self._cached_logits = logits
        self._cached_probs = probs
        self._cached_x0 = x0
        self._current_block_length = params.block_length

        # Step 3: Select tokens to decode
        x = self.token_selector.select_tokens_to_decode(
            x, x0, probs, params.remasking_strategy, tokens_to_transfer, block_start, block_end
        )

        # Update state
        self.current_state.step += 1
        self.current_state.tokens = x[0].tolist()

        # Update positions with candidates
        self._update_positions_with_candidates(logits, probs, params.top_k, tokenizer)

        # Cache state
        self.state_cache[self.current_state.step] = self.current_state

        # Check if generation is complete
        is_complete = all(not pos.is_masked or pos.position < prompt_length
                         for pos in self.current_state.positions)

        return self.current_state, is_complete

    def select_tokens_only(
        self,
        tokenizer,
        tokens_to_select: int = 1,
        remasking_strategy: str = "low_confidence",
        manual_selections: Optional[Dict[int, int]] = None
    ) -> Tuple[GenerationState, bool]:
        """Perform only token selection without forward pass using cached logits"""
        if not self.current_state:
            raise ValueError("No generation initialized")

        if not hasattr(self, '_cached_logits') or self._cached_logits is None:
            raise ValueError("No cached logits available. Run forward pass first.")

        x = torch.tensor([self.current_state.tokens]).to(self.device)
        prompt_length = self.current_state.prompt_length
        gen_length = len(self.current_state.tokens) - prompt_length

        # Apply manual selections if provided
        if manual_selections:
            for pos, token_id in manual_selections.items():
                if pos < len(self.current_state.tokens):
                    x[0, pos] = token_id
                    self.current_state.selected_positions[pos] = token_id

        # Use cached data
        probs = self._cached_probs
        x0 = self._cached_x0

        # Find which block we're working on
        current_block = self.current_state.block
        block_length = getattr(self, '_current_block_length', 32)
        num_blocks = gen_length // block_length

        if current_block >= num_blocks:
            return self.current_state, True

        # Get current block boundaries
        block_start = prompt_length + current_block * block_length
        block_end = prompt_length + (current_block + 1) * block_length

        # Check if current block is complete
        block_mask_index = (x[:, block_start:block_end] == self.mask_id)
        if block_mask_index.sum() == 0:
            self.current_state.block += 1
            return self.select_tokens_only(tokenizer, tokens_to_select, remasking_strategy, manual_selections)

        # Use the specified number of tokens to select
        tokens_to_transfer = min(tokens_to_select, block_mask_index.sum().item())

        # Perform token selection using cached data
        x_new = self.token_selector.select_tokens_to_decode(
            x, x0, probs, remasking_strategy, tokens_to_transfer, block_start, block_end
        )

        # Update state
        self.current_state.step += 1
        self.current_state.tokens = x_new[0].tolist()

        # Update positions (without recalculating candidates)
        for i, pos_data in enumerate(self.current_state.positions):
            if i < len(self.current_state.tokens):
                pos_data.current_token_id = self.current_state.tokens[i]
                pos_data.current_token = tokenizer.decode([pos_data.current_token_id])
                pos_data.is_masked = (pos_data.current_token_id == self.mask_id)

        # Cache updated state
        self.state_cache[self.current_state.step] = self.current_state

        # Check if generation is complete
        is_complete = all(not pos.is_masked or pos.position < prompt_length
                         for pos in self.current_state.positions)

        return self.current_state, is_complete

    def _update_positions_with_candidates(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        top_k: int,
        tokenizer
    ):
        """Update position data with token candidates"""
        for i, pos_data in enumerate(self.current_state.positions):
            if i < len(self.current_state.tokens):
                pos_data.current_token_id = self.current_state.tokens[i]
                pos_data.current_token = tokenizer.decode([pos_data.current_token_id])
                pos_data.is_masked = (pos_data.current_token_id == self.mask_id)

                # Get top-k candidates for this position
                if i < logits.shape[1]:
                    pos_logits = logits[0, i]
                    pos_probs = probs[0, i]
                    candidates = self.prob_processor.get_token_candidates(
                        pos_logits, pos_probs, top_k, tokenizer
                    )
                    pos_data.candidates = candidates

    def rewind_to_step(self, step: int) -> Optional[GenerationState]:
        """Rewind generation to a previous step"""
        if step in self.state_cache:
            self.current_state = self.state_cache[step]
            return self.current_state
        return None

    def get_num_transfer_tokens(self, mask_index, steps):
        """Calculate number of tokens to transfer per step"""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1
        return num_transfer_tokens