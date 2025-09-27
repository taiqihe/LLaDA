from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TokenCandidate:
    token: str
    token_id: int
    logit: float
    prob: float
    rank: int


@dataclass
class PositionData:
    is_prompt: bool
    is_masked: bool
    is_block_boundary: bool
    candidates: List[TokenCandidate]


@dataclass
class GenerationState:
    step: int
    token_ids: List[int]
    tokens: List[str]
    positions: List[PositionData]
    prompt_length: int
    block_start: int
    block_length: int
    selected_positions: Dict[int, int]  # position -> token_id


# @dataclass
# class GenerationParams:
#     tokens_to_select: int = 1
#     block_length: int = 32
#     remasking_strategy: str = "low_confidence"
#     top_k: int = 50
#     manual_selections: Optional[Dict[int, int]] = None


@dataclass
class ProbabilitySettings:
    softmax_temperature: float = 1.0
    gumbel_temperature: float = 0.0
    apply_gumbel_noise: bool = False
    top_k: int = 50
    top_p: float = 1.0


# WebSocket message types
class MessageTypes:
    MODEL_STATUS = "model_status"
    LOAD_MODEL = "load_model"
    MODEL_LOAD_RESULT = "model_load_result"
    INITIALIZE = "initialize"
    APPLY_SELECTION = "apply_selection"
    AUTO_SELECT = "auto_select"
    STATE_UPDATE = "state_update"
    FORWARD_PASS = "forward_pass"
    FORWARD_PASS_RESULT = "forward_pass_result"
    REPROCESS_PROBABILITIES = "reprocess_probabilities"
    REPROCESSED_PROBABILITIES_RESULT = "reprocessed_probabilities_result"
    REWIND = "rewind"
    ERROR = "error"