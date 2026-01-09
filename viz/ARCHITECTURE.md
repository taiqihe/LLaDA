# LLaDA Visualizer Architecture

This document describes the modular architecture of the LLaDA Diffusion Language Model Visualizer, detailing the design decisions, component relationships, and data flow.

## 🏗️ Architecture Overview

The visualizer follows a **layered, modular architecture** with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│  FastAPI Server │
│   (Browser)     │    │   (main.py)     │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ DiffusionVisualizer │
                    │   (Coordinator)     │
                    └─────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ DiffusionModel  │ │  TokenTracker   │ │WebSocketHandler │
    │  (Model Ops &   │ │  (State Mgmt &  │ │  (Communication │
    │   Caching)      │ │   Selection)    │ │   & Validation) │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
                │               │
                ▼               ▼
    ┌─────────────────┐ ┌─────────────────┐
    │ProbabilityProc. │ │  TokenSelector  │
    │ (Restrictions & │ │ (Ranking Strats)│
    │  Calculations)  │ │                 │
    └─────────────────┘ └─────────────────┘
                │               │
                ▼               ▼
    ┌─────────────────┐ ┌─────────────────┐
    │ Logger Config   │ │  Config & Models│
    │ (Monitoring)    │ │  (Data Types)   │
    └─────────────────┘ └─────────────────┘
```

## 📦 Core Components

### 1. **main.py** - Application Entry Point
**Responsibility**: Server initialization and component coordination

```python
class DiffusionVisualizer:
    def __init__(self, config: Dict = None):
        self.diffusion_model = DiffusionModel(device=config["device"], mask_id=MASK_ID)
        self.token_tracker = TokenTracker(device=config["device"], mask_id=MASK_ID)
        self.message_handler = WebSocketMessageHandler(
            self.diffusion_model, self.token_tracker, config
        )
```

**Key Features**:
- Streamlined server setup with configuration-driven initialization
- Component orchestration and dependency injection
- FastAPI WebSocket endpoint management
- CLI argument parsing for model path, host, and port

---

### 2. **models.py** - Data Structures
**Responsibility**: Type definitions and data classes

```python
@dataclass
class TokenCandidate:
    token: str
    token_id: int
    logit: float
    prob: float
    rank: int

@dataclass
class GenerationState:
    step: int
    block: int
    tokens: List[int]
    positions: List[PositionData]
    prompt_length: int
    selected_positions: Dict[int, int]
```

**Key Features**:
- Immutable data structures using dataclasses
- Type safety with annotations
- Clear parameter groupings
- JSON serialization support

---

### 3. **config.py** - Configuration Management
**Responsibility**: Centralized constants and settings

```python
# Model defaults
DEFAULT_MASK_ID = 126336
DEFAULT_DEVICE = "cuda"

# Generation defaults
DEFAULT_GEN_LENGTH = 128
DEFAULT_BLOCK_LENGTH = 32

# WebSocket message types
class MessageTypes:
    MODEL_STATUS = "model_status"
    INITIALIZE = "initialize"
    # ...
```

**Key Features**:
- Single source of truth for constants
- Environment-based configuration
- Message type enumeration
- Validation parameters

---

### 4. **diffusion_model.py** - Model Operations & Caching
**Responsibility**: Model loading, inference, and probability caching

```python
class ModelLoader:
    def load_model(self, model_path: str) -> Tuple[Model, Tokenizer, bool]

class DiffusionModel:
    def run_forward_pass(self, x: torch.Tensor) -> torch.Tensor
    def forward_pass_with_prompt(self, prompt_tokens, visual_top_k, actual_top_k, top_p) -> Dict
    def tokenize_prompt(self, prompt: str) -> List[int]
    def has_cached_results(self) -> bool
    def get_cached_results(self) -> Optional[Dict]
    def clear_cache(self)
```

**Key Features**:
- Separate model loading logic with performance logging
- **Probability Caching**: Stores logits, probs, x0 for auto_select functionality
- **Dual Restrictions**: Supports visual vs actual top_k limits
- Device management (CPU/CUDA)
- Enhanced error handling and validation

---

### 5. **probability_processor.py** - Probability Calculations & Restrictions
**Responsibility**: Logit processing, filtering, and token restrictions

```python
class ProbabilityProcessor:
    def logits_to_probabilities(self, logits, temp, gumbel_temp) -> Tuple
    def apply_top_p_filtering(self, probs: torch.Tensor, top_p: float) -> torch.Tensor
    def apply_token_restrictions(self, probs, top_k, top_p) -> Tuple[torch.Tensor, int]
    def get_token_candidates_with_restrictions(self, logits, probs, visual_top_k, actual_top_k, top_p, tokenizer) -> Tuple[List[TokenCandidate], int]
    def add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor
```

**Key Features**:
- **Smart Restrictions**: Combines top_k and top_p constraints (returns minimum)
- **Dual Token Limits**: Separate visual and actual candidate counts
- **Enhanced Candidates**: TokenCandidate.is_in_actual field for restriction tracking
- Temperature scaling and nucleus (top-p) sampling
- Numerical stability and error handling

---

### 6. **token_tracker.py** - State Management & Token Selection
**Responsibility**: Generation state tracking and intelligent token selection

```python
class TokenSelector:
    def apply_ranking_strategy(self, probs: torch.Tensor, strategy: str) -> torch.Tensor
    def _ranking_low_confidence(self, probs) -> torch.Tensor
    def _ranking_high_confidence(self, probs) -> torch.Tensor
    def _ranking_low_entropy(self, probs) -> torch.Tensor
    def _ranking_high_entropy(self, probs) -> torch.Tensor
    def _ranking_random(self, probs) -> torch.Tensor

class TokenTracker:
    def initialize_generation(self, prompt_tokens, gen_length, block_length, tokenizer) -> GenerationState
    def auto_select(self, probabilities, x0, strategy, max_tokens, selection_method) -> Dict[int, int]
    def apply_selections(self, selections: Dict[int, int], tokenizer) -> GenerationState
    def rewind_to_step(self, step: int) -> Optional[GenerationState]
```

**Key Features**:
- **Separated Concerns**: Selection logic separate from application
- **Multiple Ranking Strategies**: Low/high confidence, entropy, random
- **Cache-Aware Selection**: Works with DiffusionModel's cached probabilities
- **State Management**: Caching, rewind functionality, block-based generation
- **Flexible Selection**: Auto-select + manual override capability

---

### 7. **websocket_handlers.py** - Communication Layer & Validation
**Responsibility**: WebSocket message routing, validation, and error handling

```python
class WebSocketMessageHandler:
    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any])
    async def _handle_load_model(self, websocket, message)
    async def _handle_initialize(self, websocket, message)
    async def _handle_auto_select(self, websocket, message)
    async def _handle_apply_selection(self, websocket, message)
    async def _handle_forward_pass(self, websocket, message)
    async def _handle_reprocess_probabilities(self, websocket, message)
    async def _handle_rewind(self, websocket, message)
```

**Key Features**:
- **Comprehensive Validation**: Input validation for all message parameters
- **Enhanced Error Handling**: Detailed error messages and recovery guidance
- **New Message Types**: Support for auto_select, apply_selection workflow
- **Parameter Validation**: Range checking, type validation, and constraint enforcement
- **Structured Logging**: Integration with component-specific loggers
- Async/await support

---

### 8. **logger_config.py** - Monitoring & Debugging
**Responsibility**: Structured logging and monitoring across all components

```python
def setup_logger(name, level, log_file, console) -> logging.Logger
def get_logger(name) -> logging.Logger

# Component-specific loggers
main_logger = setup_logger("llada_visualizer", "INFO", "logs/visualizer.log", True)
diffusion_logger = setup_logger("llada_visualizer.diffusion", "DEBUG", "logs/diffusion.log", False)
token_tracker_logger = setup_logger("llada_visualizer.token_tracker", "DEBUG", "logs/token_tracker.log", False)
websocket_logger = setup_logger("llada_visualizer.websocket", "INFO", "logs/websocket.log", False)
probability_logger = setup_logger("llada_visualizer.probability", "DEBUG", "logs/probability.log", False)
```

**Key Features**:
- **Component-Specific Logging**: Separate loggers for each major component
- **Configurable Levels**: Debug, info, warning, error levels per component
- **File & Console Output**: Structured logging to files with console output for main events
- **Performance Monitoring**: Model parameter counts, tensor shapes, operation timing
- **Debugging Support**: Detailed logs for troubleshooting and development

---

## 🔄 New Data Flow (TokenTracker Architecture)

### 1. **Model Loading Flow**
```
Client Request → WebSocketHandler → DiffusionModel → ModelLoader → Logger → Response
```

### 2. **Generation Initialization Flow**
```
Prompt → DiffusionModel.tokenize_prompt() → TokenTracker.initialize_generation() → GenerationState
```

### 3. **New Multi-Step Generation Flow (Simplified)**
```
1. Initialize:
   Client → WebSocketHandler → TokenTracker.initialize_generation()
   ├── Tokenize prompt
   ├── Create masked sequence (gen_length tokens)
   └── Response with initial state

2. Forward Pass (with Loading Indicator):
   Client → WebSocketHandler → DiffusionModel.forward_pass_with_prompt()
   ├── Show loading indicator in UI
   ├── Extract prompt from current state
   ├── Run model inference with gen_length
   ├── ProbabilityProcessor.get_token_candidates_with_restrictions()
   ├── Cache: logits, probs, x0, tokens
   ├── Hide loading indicator
   └── Response with positions + candidates + dual restrictions

3. Auto Selection (optional):
   Client → WebSocketHandler → TokenTracker.auto_select()
   ├── Use cached probabilities from DiffusionModel
   ├── TokenSelector.apply_ranking_strategy()
   ├── Select top tokens based on strategy
   └── Response with selected_tokens

4. Apply Selection:
   Client → WebSocketHandler → TokenTracker.apply_selections()
   ├── Update GenerationState with selections (auto or manual)
   ├── Update tokens and increment step
   └── Response with updated state

5. Repeat steps 2-4 until generation is complete
```

### 4. **Probability Reprocessing Flow (Enhanced)**
```
Raw Logits → ProbabilityProcessor.reprocess_probabilities_with_settings()
├── Apply visual_top_k vs actual_top_k restrictions
├── Enhanced candidates with is_in_actual field
└── Response with dual restriction information
```

## 🎯 Design Patterns

### 1. **Dependency Injection**
Components receive their dependencies through constructor injection:

```python
class WebSocketMessageHandler:
    def __init__(self, diffusion_model: DiffusionModel, token_tracker: TokenTracker, config: Dict):
        self.diffusion_model = diffusion_model
        self.token_tracker = token_tracker
        self.config = config
```

### 2. **Strategy Pattern**
Token ranking strategies are implemented as pluggable algorithms:

```python
def apply_ranking_strategy(self, probs: torch.Tensor, strategy: str) -> torch.Tensor:
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
            return self._ranking_low_confidence(probs)
```

### 3. **Command Pattern**
WebSocket messages are handled as discrete commands:

```python
match message_type:
    case MessageTypes.LOAD_MODEL:
        await self._handle_load_model(websocket, message)
    case MessageTypes.INITIALIZE:
        await self._handle_initialize(websocket, message)
    case MessageTypes.AUTO_SELECT:
        await self._handle_auto_select(websocket, message)
    case MessageTypes.APPLY_SELECTION:
        await self._handle_apply_selection(websocket, message)
    case MessageTypes.FORWARD_PASS:
        await self._handle_forward_pass(websocket, message)
```

### 4. **State Pattern**
Generation progresses through discrete states with caching:

```python
self.state_cache[step] = current_state  # State caching
current_state = self.state_cache[step]  # State rewind
```

## 🚀 Performance Optimizations

### 1. **Enhanced Caching Strategy**
- **Probability Caching**: DiffusionModel caches logits, probs, x0, tokens after forward pass
- **Auto-Select Optimization**: TokenTracker uses cached results instead of re-running forward pass
- **State Caching**: Store generation states for rewind functionality
- **Model Caching**: Keep model loaded in memory with performance logging
- **UI Optimization**: Loading indicators provide feedback during long-running operations

### 2. **Lazy Loading**
- Models loaded only when needed
- Components initialized on-demand
- Progressive state updates

### 3. **Memory Management**
- Tensor operations use appropriate dtypes (bfloat16)
- Device management (CPU/CUDA) based on availability
- Gradient tracking disabled during inference

### 4. **Async Processing**
- WebSocket handlers use async/await
- Non-blocking message processing
- Concurrent client support

## 🔧 Extensibility Points

### 1. **Adding New Remasking Strategies**
```python
def _remasking_custom_strategy(self, x0: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    # Custom logic here
    return confidence_scores

# Register in apply_remasking_strategy()
```

### 2. **Adding New Message Types**
```python
# 1. Add to config.py
class MessageTypes:
    NEW_OPERATION = "new_operation"

# 2. Add handler in websocket_handlers.py
async def _handle_new_operation(self, websocket, message):
    # Handler logic

# 3. Register in handle_message()
```

### 3. **Custom Probability Processors**
```python
class CustomProbabilityProcessor(ProbabilityProcessor):
    def apply_custom_filtering(self, probs: torch.Tensor) -> torch.Tensor:
        # Custom filtering logic
        return filtered_probs
```

## 📊 Metrics and Monitoring

### 1. **Performance Metrics**
- Model loading time
- Forward pass latency
- WebSocket response time
- Memory usage tracking

### 2. **Error Tracking**
- Model loading failures
- Generation errors
- WebSocket disconnections
- Invalid message handling

### 3. **Usage Analytics**
- Active connections
- Generation requests per session
- Popular remasking strategies
- Average generation length

## 🔒 Security Considerations

### 1. **Input Validation**
- Message type validation
- Parameter range checking
- Model path sanitization
- Token sequence validation

### 2. **Resource Limits**
- Maximum generation length
- Model size constraints
- Memory usage limits
- Connection timeouts

### 3. **Error Handling**
- Graceful degradation
- Error message sanitization
- Resource cleanup on errors
- Connection state management

## 🧪 Testing Strategy

### 1. **Unit Tests**
- Component isolation testing
- Mock external dependencies
- Edge case validation
- Error condition testing

### 2. **Integration Tests**
- End-to-end message flow
- WebSocket communication
- Model loading scenarios
- State management validation

### 3. **Performance Tests**
- Load testing with multiple clients
- Memory leak detection
- GPU utilization monitoring
- Response time benchmarking

This modular architecture enables maintainable, testable, and extensible code while providing clear separation of concerns and robust error handling.