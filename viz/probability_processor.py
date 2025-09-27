import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

from config import EPSILON, MIN_TEMPERATURE
from models import TokenCandidate


class ProbabilityProcessor:
    """Handles probability calculations, temperature scaling, and token filtering."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def logits_to_probabilities(
        self,
        logits: torch.Tensor,
        softmax_temperature: float = 1.0,
        gumbel_temperature: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert logits to probabilities and apply temperature/noise"""
        # Apply softmax temperature
        scaled_logits = logits / softmax_temperature

        # Apply Gumbel noise if gumbel_temperature > 0
        if gumbel_temperature > 0:
            logits_noisy = self.add_gumbel_noise(scaled_logits, gumbel_temperature)
        else:
            logits_noisy = scaled_logits

        # Get predicted tokens (x0)
        x0 = torch.argmax(logits_noisy, dim=-1)

        # Get probabilities from scaled logits (for remasking)
        probs = F.softmax(scaled_logits, dim=-1)

        return x0, probs

    def add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply Gumbel noise for sampling"""
        if temperature == 0:
            return logits
        logits = logits.to(torch.float)
        noise = torch.rand_like(logits, dtype=torch.float)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def apply_top_p_filtering(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) sampling by zeroing out low probability tokens"""
        if top_p >= 1.0:
            return probs

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Get cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff - tokens to remove (where cumsum > top_p)
        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least the first token (highest prob)
        sorted_indices_to_remove[..., 0] = False

        # Create mask for original indices
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

        # Zero out the probabilities and renormalize
        filtered_probs = probs.clone()
        filtered_probs[indices_to_remove] = 0.0

        # Avoid division by zero
        prob_sums = filtered_probs.sum(dim=-1, keepdim=True)
        prob_sums = torch.clamp(prob_sums, min=EPSILON)
        filtered_probs = filtered_probs / prob_sums

        return filtered_probs

    def apply_token_restrictions(self, probs: torch.Tensor, top_k: int, top_p: float) -> Tuple[torch.Tensor, int]:
        """Apply both top_k and top_p restrictions, returning the more restrictive count"""
        # Apply top_p filtering first
        filtered_probs = self.apply_top_p_filtering(probs, top_p)

        # Count how many tokens remain after top_p filtering
        top_p_count = (filtered_probs > 0).sum(dim=-1)

        # The actual restriction is the minimum of top_k and top_p count
        actual_top_k = torch.minimum(torch.tensor(top_k), top_p_count)

        return filtered_probs, actual_top_k.item()

    def get_token_candidates_with_restrictions(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        visual_top_k: int,
        actual_top_k: int,
        top_p: float,
        tokenizer
    ) -> Tuple[List[TokenCandidate], int]:
        """Get token candidates with both visual and actual restrictions"""
        # Check for valid probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return [], 0

        # Apply restrictions to get actual constraints
        _, actual_restricted_k = self.apply_token_restrictions(probs, actual_top_k, top_p)

        # For visual display, use the larger of visual_top_k or actual_restricted_k
        display_k = max(visual_top_k, actual_restricted_k)
        valid_k = min(display_k, probs.shape[0])

        # Get top candidates for display
        top_k_indices = torch.topk(probs, k=valid_k).indices

        candidates = []
        for rank, idx in enumerate(top_k_indices):
            token_id = idx.item()
            try:
                token_text = tokenizer.decode([token_id])
            except:
                token_text = f"<token_{token_id}>"

            # Mark whether this token is within actual restrictions
            is_in_actual = rank < actual_restricted_k

            candidates.append(TokenCandidate(
                token=token_text,
                token_id=token_id,
                logit=float(logits[idx].item()),
                prob=float(probs[idx].item()),
                rank=rank,
                is_in_actual=is_in_actual
            ))

        return candidates, actual_restricted_k

    def get_token_candidates(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        top_k: int,
        tokenizer
    ) -> List[TokenCandidate]:
        """Get top-k token candidates for a position"""
        # Check for valid probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return []

        # Get top-k candidates (after any filtering)
        valid_k = min(top_k, probs.shape[0])
        top_k_indices = torch.topk(probs, k=valid_k).indices

        candidates = []
        for rank, idx in enumerate(top_k_indices):
            token_id = idx.item()
            try:
                token_text = tokenizer.decode([token_id])
            except:
                token_text = f"<token_{token_id}>"

            candidates.append(TokenCandidate(
                token=token_text,
                token_id=token_id,
                logit=float(logits[idx].item()),
                prob=float(probs[idx].item()),
                rank=rank
            ))

        return candidates

    def reprocess_probabilities_with_settings(
        self,
        raw_logits: List[List[float]],
        settings: Dict,
        tokenizer
    ) -> Dict:
        """Reprocess raw logits with new probability settings"""
        try:
            # Handle new sparse logits format
            if not raw_logits or not isinstance(raw_logits, list):
                raise ValueError("Invalid raw_logits format")

            print(f"Received sparse logits for {len(raw_logits)} positions")

            # For sparse logits, we'll work with them directly without converting to full tensors
            # This avoids memory issues and message size problems

            # Apply new settings with validation
            softmax_temp = max(MIN_TEMPERATURE, float(settings.get('softmax_temperature', 1.0)))
            top_k = max(1, int(settings.get('top_k', 3)))  # Limit to 3 for message size

            print(f"Settings: softmax_temp={softmax_temp}, top_k={top_k}")

            # Process sparse logits
            positions_data = []
            for i, pos_sparse_logits in enumerate(raw_logits[:5]):  # Limit to 5 positions
                if not isinstance(pos_sparse_logits, dict):
                    continue

                # Convert sparse logits to probabilities
                token_ids = list(pos_sparse_logits.keys())
                logit_values = list(pos_sparse_logits.values())

                if not token_ids:
                    continue

                # Apply temperature scaling
                scaled_logits = [logit / softmax_temp for logit in logit_values]

                # Convert to probabilities
                max_logit = max(scaled_logits)
                exp_logits = [math.exp(logit - max_logit) for logit in scaled_logits]
                sum_exp = sum(exp_logits)
                probs = [exp_logit / sum_exp for exp_logit in exp_logits]

                # Create candidates
                candidates_data = []
                for j, (token_id, prob) in enumerate(zip(token_ids, probs)):
                    if j >= top_k:  # Limit candidates
                        break

                    try:
                        token_text = tokenizer.decode([int(token_id)])
                    except:
                        token_text = f"<token_{token_id}>"

                    candidates_data.append({
                        'token': token_text[:10],  # Truncate token text
                        'token_id': int(token_id),
                        'logit': round(scaled_logits[j], 2),
                        'prob': round(prob, 3),
                        'rank': j
                    })

                positions_data.append({
                    'position': i,
                    'candidates': candidates_data
                })

            return {
                'positions': positions_data
            }

        except Exception as e:
            print(f"Error in reprocess_probabilities_with_settings: {e}")
            import traceback
            traceback.print_exc()
            raise