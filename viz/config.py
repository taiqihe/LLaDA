"""Configuration constants for the diffusion language model visualizer."""

# Default model settings
DEFAULT_MASK_ID = 126336
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float"

# Generation defaults
DEFAULT_GEN_LENGTH = 128
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_TOKENS_TO_SELECT = 1
DEFAULT_TOP_K = 10
DEFAULT_TOP_P = 1.0

# Temperature settings
DEFAULT_SOFTMAX_TEMPERATURE = 1.0
DEFAULT_GUMBEL_TEMPERATURE = 0.0

# Remasking strategies
REMASKING_STRATEGIES = [
    "low_confidence",
    "high_confidence",
    "low_entropy",
    "high_entropy",
    "random"
]

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Numerical stability
EPSILON = 1e-10
MIN_TEMPERATURE = 0.1
MIN_TOP_P = 0.01

# WebSocket message types
class MessageTypes:
    MODEL_STATUS = "model_status"
    LOAD_MODEL = "load_model"
    MODEL_LOAD_RESULT = "model_load_result"
    INITIALIZE = "initialize"
    STEP = "step"
    STATE_UPDATE = "state_update"
    TOKENIZE_AND_FORWARD = "tokenize_and_forward"
    FORWARD_PASS = "forward_pass"
    FORWARD_PASS_RESULT = "forward_pass_result"
    SELECT_TOKENS_ONLY = "select_tokens_only"
    REPROCESS_PROBABILITIES = "reprocess_probabilities"
    REPROCESSED_PROBABILITIES_RESULT = "reprocessed_probabilities_result"
    REWIND = "rewind"
    ERROR = "error"