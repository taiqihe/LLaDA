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
    position: int
    current_token: Optional[str]
    current_token_id: Optional[int]
    is_masked: bool
    candidates: List[TokenCandidate]
    is_block_boundary: bool


@dataclass
class GenerationState:
    step: int
    block: int
    tokens: List[int]
    positions: List[PositionData]
    prompt_length: int
    selected_positions: Dict[int, int]  # position -> token_id


@dataclass
class GenerationParams:
    tokens_to_select: int = 1
    block_length: int = 32
    gumbel_temperature: float = 0.0
    remasking_strategy: str = "low_confidence"
    top_k: int = 10
    manual_selections: Optional[Dict[int, int]] = None


@dataclass
class ProbabilitySettings:
    softmax_temperature: float = 1.0
    gumbel_temperature: float = 0.0
    apply_gumbel_noise: bool = False
    top_k: int = 10
    top_p: float = 1.0