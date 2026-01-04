import torch
import torch.nn.functional as F
import traceback
from typing import Optional, List, Dict
from transformers import AutoModel, AutoTokenizer

from probability_processor import ProbabilityProcessor
from logger_config import diffusion_logger


class ModelLoader:
    """Handles model loading and validation."""

    def __init__(self, device: str):
        self.device = device if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str):
        """Load model and tokenizer from path"""
        try:
            from transformers import AutoTokenizer, AutoModel

            diffusion_logger.info(f"Loading model from {model_path} on {self.device}...")

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16
            ).to(self.device).eval()

            diffusion_logger.info(f"Model loaded successfully! Parameters: {sum(p.numel() for p in model.parameters()):,}")
            return model, tokenizer, True

        except Exception as e:
            diffusion_logger.error(f"Error loading model: {e}")
            traceback.print_exc()
            return None, None, False


class DiffusionModel:
    """Core diffusion model operations and forward pass logic."""

    def __init__(self, device: str, mask_id: int):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mask_id = mask_id
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.prob_processor = ProbabilityProcessor(self.device)

        self.model_loader = ModelLoader(self.device)

        # Cache for forward pass results
        self._cached_logits = None
        self._cached_probs = None
        self._cached_x0 = None
        self._cached_tokens = None
        self._cached_prompt_length = None

    def load_model(self, model_path: str) -> bool:
        """Load model from path"""
        diffusion_logger.info(f"Attempting to load model: {model_path}")
        self.model, self.tokenizer, success = self.model_loader.load_model(model_path)
        if success:
            self.model_path = model_path
            diffusion_logger.info(f"Model successfully loaded and assigned: {model_path}")
        else:
            diffusion_logger.error(f"Failed to load model: {model_path}")
        return success

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and self.tokenizer is not None

    def run_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and get raw logits"""
        if not self.is_model_loaded():
            raise ValueError("Model not loaded")

        diffusion_logger.debug(f"Running forward pass on tensor shape: {x.shape}")
        with torch.no_grad():
            logits = self.model(x).logits
        diffusion_logger.debug(f"Forward pass complete, logits shape: {logits.shape}")
        return logits

    def forward_pass_with_prompt(
        self,
        prompt_tokens: List[int],
        gen_length: int = 128,
        visual_top_k: int = 20,
        actual_top_k: int = 10,
        top_p: float = 1.0
    ) -> Dict:
        """Run a forward pass with masked tokens and return logits for all positions"""
        if not self.is_model_loaded():
            raise ValueError("Model not loaded. Please load a model first.")

        # Create sequence with prompt + masked tokens
        total_length = len(prompt_tokens) + gen_length
        x = torch.full((1, total_length), self.mask_id, dtype=torch.long).to(self.device)
        x[:, :len(prompt_tokens)] = torch.tensor(prompt_tokens).to(self.device)

        # Run forward pass
        logits = self.run_forward_pass(x)[0]  # Remove batch dimension

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get predicted tokens (x0)
        x0 = torch.argmax(logits, dim=-1)

        # Cache results for auto_select functionality
        self._cached_logits = logits.clone()
        self._cached_probs = probs.clone()
        self._cached_x0 = x0.clone()
        self._cached_tokens = x[0].clone()
        self._cached_prompt_length = len(prompt_tokens)

        # Prepare position data - focus on masked positions
        positions_data = []
        for i in range(total_length):
            if i < logits.shape[0]:  # Make sure we have logits for this position
                pos_logits = logits[i]
                pos_probs = probs[i]

                is_masked = i >= len(prompt_tokens)
                current_token_id = x[0, i].item()
                current_token = self.tokenizer.decode([current_token_id])

                # Get candidates with restrictions
                candidates, actual_restricted_k = self.prob_processor.get_token_candidates_with_restrictions(
                    pos_logits, pos_probs, visual_top_k, actual_top_k, top_p, self.tokenizer
                )

                # Convert to dict format for JSON serialization
                candidates_dict = [
                    {
                        'token': c.token,
                        'token_id': c.token_id,
                        'logit': c.logit,
                        'prob': c.prob,
                        'rank': c.rank,
                        'is_in_actual': c.is_in_actual
                    }
                    for c in candidates
                ]

                positions_data.append({
                    'position': i,
                    'current_token': current_token,
                    'current_token_id': current_token_id,
                    'is_masked': is_masked,
                    'candidates': candidates_dict,
                    'is_prompt': i < len(prompt_tokens),
                    'actual_top_k': actual_restricted_k
                })

        # Limit raw_logits to avoid massive WebSocket messages
        # Only include top-k logits for the first few positions
        limited_logits = []
        max_positions_for_logits = min(10, logits.shape[0])
        vocab_size = logits.shape[1]
        top_k_for_logits = min(50, vocab_size)  # Only top 50 logits per position

        for i in range(max_positions_for_logits):
            pos_logits = logits[i]
            # Get top-k indices and values
            top_values, top_indices = torch.topk(pos_logits, top_k_for_logits)
            # Create sparse representation
            sparse_logits = {int(idx.item()): float(val.item()) for idx, val in zip(top_indices, top_values)}
            limited_logits.append(sparse_logits)

        return {
            'positions': positions_data,
            'raw_logits': limited_logits,  # Limited sparse logits to avoid message size issues
            'tokens': x[0].cpu().tolist(),
            'prompt_length': len(prompt_tokens)
        }

    def tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize prompt with BOS token"""
        if not self.is_model_loaded():
            raise ValueError("Model not loaded")

        bos_token = self.tokenizer.get_vocab()[self.tokenizer.special_tokens_map["bos_token"]]
        input_ids = [bos_token] + self.tokenizer(prompt)["input_ids"]
        return input_ids

    def reprocess_probabilities_with_settings(self, raw_logits: List[List[float]], settings: Dict) -> Dict:
        """Reprocess raw logits with new probability settings"""
        if not self.is_model_loaded():
            raise ValueError("Model not loaded")

        return self.prob_processor.reprocess_probabilities_with_settings(
            raw_logits, settings, self.tokenizer
        )

    def has_cached_results(self) -> bool:
        """Check if forward pass results are cached"""
        return (self._cached_logits is not None and
                self._cached_probs is not None and
                self._cached_x0 is not None and
                self._cached_tokens is not None)

    def get_cached_results(self) -> Optional[Dict]:
        """Get cached forward pass results"""
        if not self.has_cached_results():
            return None

        return {
            'logits': self._cached_logits,
            'probs': self._cached_probs,
            'x0': self._cached_x0,
            'tokens': self._cached_tokens,
            'prompt_length': self._cached_prompt_length
        }

    def clear_cache(self):
        """Clear cached forward pass results"""
        self._cached_logits = None
        self._cached_probs = None
        self._cached_x0 = None
        self._cached_tokens = None
        self._cached_prompt_length = None