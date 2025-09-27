"""Configuration constants for the diffusion language model visualizer."""

MASK_ID = 126336

# Remasking strategies
REMASKING_STRATEGIES = [
    "low_confidence",
    "high_confidence",
    "low_entropy",
    "high_entropy",
    "random"
]

# Numerical stability
EPSILON = 1e-10
MIN_TEMPERATURE = 0.1
MIN_TOP_P = 0.01

DEFAULT_CONFIG = {
    "device": "cuda:0",
    "dtype": "float",
    "gen_length": 128,
    "block_length": 32,
    "tokens_to_select": 1,
    "remasking": "low_confidence",
    "selection": "constant",
    "visual_top_k": 20,  # Number of tokens to show in UI
    "actual_top_k": 10,  # Number of tokens for actual selection
    "top_p": 1.0,
    "softmax_temperature": 1.0,
    "gumbel_temperature": 0.0,
    "host": "0.0.0.0",
    "port": 8000,
}