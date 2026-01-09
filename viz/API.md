# LLaDA Visualizer WebSocket API

This document describes the WebSocket API for the LLaDA Diffusion Language Model Visualizer, including message formats, request/response patterns, and usage examples.

## 🔌 Connection

### WebSocket Endpoint
```
ws://localhost:8000/ws
```

### Connection Flow
1. Client connects to WebSocket endpoint
2. Server sends initial `model_status` message
3. Client can send commands and receive responses
4. Server maintains connection until client disconnects

## 📨 Message Format

All messages use JSON format:

```json
{
  "type": "message_type",
  "param1": "value1",
  "param2": "value2"
}
```

## 🔄 New Workflow

The LLaDA visualizer now uses a **multi-step workflow** that separates token selection from application:

1. **Initialize** - Set up generation state with prompt and parameters
2. **Forward Pass** - Run model inference and cache probabilities
3. **Auto-Select** - Use ranking strategies to select tokens (requires cached probabilities)
4. **Apply Selection** - Apply selected tokens to the generation state
5. **Repeat** - Continue with steps 2-4 until generation is complete

This workflow provides better control and allows for:
- **Dual Restrictions**: Visual top_k (for UI display) vs actual top_k (for selection)
- **Probability Caching**: Avoid re-running inference for token selection
- **Manual Override**: Combine auto-selection with manual token choices
- **Strategy Comparison**: Try different ranking strategies on the same forward pass

---

## 📋 Message Types

### 1. Model Management

#### `load_model` - Load Diffusion Model

**Request:**
```json
{
  "type": "load_model",
  "model_path": "GSAI-ML/LLaDA-8B-Base"
}
```

**Response:**
```json
{
  "type": "model_load_result",
  "success": true,
  "model_path": "GSAI-ML/LLaDA-8B-Base",
  "error": null
}
```

**Parameters:**
- `model_path` (string): HuggingFace model ID or local path

**Error Response:**
```json
{
  "type": "model_load_result",
  "success": false,
  "model_path": null,
  "error": "Failed to load model from path"
}
```

---

#### `model_status` - Model Status (Server-Initiated)

**Server Message:**
```json
{
  "type": "model_status",
  "is_loaded": true,
  "model_path": "GSAI-ML/LLaDA-8B-Base"
}
```

**Fields:**
- `is_loaded` (boolean): Whether a model is currently loaded
- `model_path` (string|null): Path/ID of loaded model

---

### 2. Generation Control

#### `initialize` - Initialize Generation

**Request:**
```json
{
  "type": "initialize",
  "prompt": "The future of artificial intelligence is",
  "gen_length": 128
}
```

**Parameters:**
- `prompt` (string, required): Input prompt text
- `gen_length` (integer, optional): Number of tokens to generate (default: 128, range: 1-1024)

**Response:**
```json
{
  "type": "state_update",
  "state": {
    "step": 0,
    "block": 0,
    "token_ids": [1, 450, 3930, 315, 21075, 6677, 374],
    "tokens": ["<s>", "The", "future", "of", "artificial", "intelligence", "is"],
    "positions": [...],
    "prompt_length": 7,
    "selected_positions": {}
  },
  "is_complete": false
}
```

**Notes:**
- Block length is automatically set to match gen_length (no separate block-based generation)
- After initialization, use `forward_pass` to run model inference

---

#### `auto_select` - Automatic Token Selection

**Request:**
```json
{
  "type": "auto_select",
  "tokens_to_select": 2,
  "strategy": "low_confidence",
  "selection": "constant"
}
```

**Response:**
```json
{
  "type": "state_update",
  "selected_tokens": {
    "15": 1234,
    "16": 5678
  },
  "message": "Auto-selected 2 tokens using low_confidence strategy."
}
```

**Parameters:**
- `tokens_to_select` (integer, optional): Number of tokens to select (default: 1)
- `strategy` (string, optional): Token ranking strategy (default: "low_confidence")
- `selection` (string, optional): Selection method (default: "constant")

**Prerequisites:**
- Must have initialized generation with `initialize`
- Must have run `forward_pass` to cache probabilities

**Ranking Strategies:**
- `"low_confidence"`: Select tokens with lowest prediction confidence
- `"high_confidence"`: Select tokens with highest prediction confidence
- `"low_entropy"`: Select positions with most certain distributions
- `"high_entropy"`: Select positions with most uncertain distributions
- `"random"`: Random token selection

---

#### `apply_selection` - Apply Token Selections

**Request:**
```json
{
  "type": "apply_selection",
  "selections": {
    "15": 1234,
    "16": 5678,
    "20": 9876
  }
}
```

**Response:**
```json
{
  "type": "state_update",
  "state": {
    "step": 1,
    "token_ids": [1, 450, 3930, 315, 21075, 6677, 374, 1234, 5678, ...],
    "tokens": ["<s>", "The", "future", "of", "AI", "is", "bright", "amazing", "powerful", ...],
    "positions": [...],
    "prompt_length": 7,
    "block_start": 7,
    "block_length": 32,
    "selected_positions": {"15": 1234, "16": 5678, "20": 9876}
  }
}
```

**Parameters:**
- `selections` (object, required): Dictionary of position -> token_id mappings

---


### 3. Probability Analysis

#### `tokenize_and_forward` - Tokenize and Run Forward Pass

**Request:**
```json
{
  "type": "tokenize_and_forward",
  "prompt": "Hello world",
  "gen_length": 64,
  "top_k": 20
}
```

**Response:**
```json
{
  "type": "forward_pass_result",
  "result": {
    "positions": [
      {
        "position": 0,
        "current_token": "<s>",
        "current_token_id": 1,
        "is_masked": false,
        "candidates": [...],
        "is_prompt": true
      },
      ...
    ],
    "raw_logits": [[...], [...], ...],
    "tokens": [1, 450, 126336, 126336, ...],
    "prompt_length": 2
  }
}
```

**Parameters:**
- `prompt` (string): Text to tokenize and process
- `gen_length` (integer, optional): Length of generation sequence (default: 128)
- `top_k` (integer, optional): Number of top candidates (default: 10)

---

#### `forward_pass` - Run Forward Pass on Current State

**Request:**
```json
{
  "type": "forward_pass",
  "gen_length": 128,
  "visual_top_k": 20,
  "actual_top_k": 10,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "type": "forward_pass_result",
  "result": {
    "positions": [
      {
        "position": 0,
        "current_token": "<s>",
        "current_token_id": 1,
        "is_masked": false,
        "candidates": [
          {
            "token": " amazing",
            "token_id": 1234,
            "logit": 3.45,
            "prob": 0.85,
            "rank": 0,
            "is_in_actual": true
          }
        ],
        "is_prompt": true,
        "actual_top_k": 10
      }
    ],
    "raw_logits": [...],
    "tokens": [1, 450, 3930, 126336, 126336],
    "prompt_length": 3
  }
}
```

**Parameters:**
- `gen_length` (integer, optional): Number of tokens to generate (default: 128, uses value from initialize if not specified)
- `visual_top_k` (integer, optional): Number of candidates to show in UI (default: 20, range: 1-100)
- `actual_top_k` (integer, optional): Number of candidates for actual selection (default: 10, range: 1-100)
- `top_p` (float, optional): Nucleus sampling threshold (default: 1.0, range: 0.01-1.0)

**Prerequisites:**
- Must have initialized generation with `initialize`
- Shows loading indicator in UI during execution

**Notes:**
- Caches probabilities for use with `auto_select`
- Uses prompt from current generation state
- Visual top_k determines display, actual top_k determines selectable candidates

---

#### `reprocess_probabilities` - Reprocess with New Settings

**Request:**
```json
{
  "type": "reprocess_probabilities",
  "raw_logits": [[...], [...], ...],
  "settings": {
    "softmax_temperature": 0.8,
    "gumbel_temperature": 0.1,
    "apply_gumbel_noise": true,
    "visual_top_k": 25,
    "actual_top_k": 15,
    "top_p": 0.9
  }
}
```

**Response:**
```json
{
  "type": "reprocessed_probabilities_result",
  "result": {
    "positions": [
      {
        "position": 0,
        "candidates": [
          {
            "token": " amazing",
            "token_id": 1234,
            "logit": 3.45,
            "prob": 0.85,
            "rank": 0,
            "is_in_actual": true
          },
          {
            "token": " incredible",
            "token_id": 5678,
            "logit": 3.12,
            "prob": 0.73,
            "rank": 1,
            "is_in_actual": true
          },
          ...
        ]
      },
      ...
    ],
    "settings_applied": { ... }
  }
}
```

**Parameters:**
- `raw_logits` (array): Raw logit values from forward pass (sparse format)
- `settings` (object): Probability processing settings
  - `softmax_temperature` (float): Temperature for softmax (default: 1.0, range: 0.1-5.0)
  - `gumbel_temperature` (float): Gumbel noise temperature (default: 0.0, range: 0.0-2.0)
  - `apply_gumbel_noise` (boolean): Whether to apply Gumbel noise (default: false)
  - `visual_top_k` (integer): Number of candidates to show in UI (default: 20, range: 1-100)
  - `actual_top_k` (integer): Number of candidates for selection (default: 10, range: 1-100)
  - `top_p` (float): Nucleus sampling threshold (default: 1.0, range: 0.01-1.0)

**Notes:**
- Processes all positions in raw_logits (not limited)
- Each candidate includes `is_in_actual` field to indicate if it's within actual restrictions
- Allows dynamic adjustment without re-running model inference

---

### 4. State Management

#### `rewind` - Rewind to Previous Step

**Request:**
```json
{
  "type": "rewind",
  "step": 3
}
```

**Response:**
```json
{
  "type": "state_update",
  "state": { ... },
  "is_complete": false
}
```

**Parameters:**
- `step` (integer): Step number to rewind to

---

### 5. Error Handling

#### `error` - Error Response

**Server Message:**
```json
{
  "type": "error",
  "message": "Model not loaded. Please load a model first."
}
```

**Common Error Messages:**
- `"Model not loaded. Please load a model first."`
- `"No generation initialized"`
- `"Invalid raw_logits format"`
- `"No cached logits available. Run forward pass first."`
- `"Unknown message type: invalid_type"`

---

## 📊 Data Structures

### GenerationState

```typescript
interface GenerationState {
  step: number;                    // Current generation step
  token_ids: number[];             // Token ID sequence
  tokens: string[];                // Token text sequence
  positions: PositionData[];       // Position information
  prompt_length: number;           // Length of input prompt
  block_start: number;             // Current block starting position
  block_length: number;            // Block size for generation
  selected_positions: {[pos: number]: number}; // Applied selections
}
```

### PositionData

```typescript
interface PositionData {
  is_prompt: boolean;              // Whether position is part of prompt
  is_masked: boolean;              // Whether position is masked
  is_block_boundary: boolean;      // Whether position starts a block
  candidates: TokenCandidate[];    // Top-k candidates
}
```

### TokenCandidate

```typescript
interface TokenCandidate {
  token: string;                   // Token text
  token_id: number;                // Token ID
  logit: number;                   // Raw logit value
  prob: number;                    // Probability after processing
  rank: number;                    // Rank in top-k list
  is_in_actual: boolean;           // Whether within actual restrictions
}
```

### Forward Pass Result

```typescript
interface ForwardPassResult {
  positions: Array<{
    position: number;              // Position in sequence
    current_token: string;         // Current token text
    current_token_id: number;      // Current token ID
    is_masked: boolean;            // Whether position is masked
    candidates: TokenCandidate[];  // Candidate tokens
    is_prompt: boolean;            // Whether part of prompt
    actual_top_k: number;          // Effective restriction count
  }>;
  raw_logits: object[];            // Sparse logits (limited for performance)
  tokens: number[];                // Full token sequence
  prompt_length: number;           // Length of prompt
}
```

---

## 🔄 Usage Examples

### Example 1: New Generation Flow

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// 1. Wait for model status and initialize
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'model_status' && msg.is_loaded) {
    // 2. Initialize generation state
    ws.send(JSON.stringify({
      type: 'initialize',
      prompt: 'The future of AI is',
      gen_length: 64,
      block_length: 16
    }));
  }

  if (msg.type === 'state_update' && msg.state) {
    // 3. Run forward pass to get probabilities
    ws.send(JSON.stringify({
      type: 'forward_pass',
      gen_length: 64,
      visual_top_k: 20,
      actual_top_k: 10,
      top_p: 0.9
    }));
  }

  if (msg.type === 'forward_pass_result') {
    // 4. Auto-select tokens based on cached probabilities
    ws.send(JSON.stringify({
      type: 'auto_select',
      tokens_to_select: 2,
      strategy: 'low_confidence'
    }));
  }

  if (msg.type === 'state_update' && msg.selected_tokens) {
    // 5. Apply the selected tokens
    ws.send(JSON.stringify({
      type: 'apply_selection',
      selections: msg.selected_tokens
    }));
  }
};
```

### Example 2: Probability Analysis with Restrictions

```javascript
// Analyze probabilities with dual restrictions
ws.send(JSON.stringify({
  type: 'forward_pass',
  gen_length: 64,
  visual_top_k: 30,    // Show 30 candidates in UI
  actual_top_k: 15,    // Only 15 available for selection
  top_p: 0.85          // Further restrict by nucleus sampling
}));

// Analyze the results
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'forward_pass_result') {
    // Check which tokens are within actual restrictions
    msg.result.positions.forEach(pos => {
      const actualCandidates = pos.candidates.filter(c => c.is_in_actual);
      console.log(`Position ${pos.position}: ${actualCandidates.length}/${pos.candidates.length} candidates available for selection`);
    });

    // Reprocess with different settings
    ws.send(JSON.stringify({
      type: 'reprocess_probabilities',
      raw_logits: msg.result.raw_logits,
      settings: {
        softmax_temperature: 0.5,
        top_p: 0.8,
        visual_top_k: 20,
        actual_top_k: 10
      }
    }));
  }
};
```

### Example 3: Manual Token Selection

```javascript
// Apply manual token selections directly
ws.send(JSON.stringify({
  type: 'apply_selection',
  selections: {
    "10": 1234,  // Force token 1234 at position 10
    "11": 5678,  // Force token 5678 at position 11
    "12": 9012   // Force token 9012 at position 12
  }
}));

// Or combine with auto-selection
ws.send(JSON.stringify({
  type: 'auto_select',
  tokens_to_select: 3,
  strategy: 'low_entropy'
}));

// Then apply the auto-selected tokens
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'state_update' && msg.selected_tokens) {
    // Optionally modify the auto-selections before applying
    const modifiedSelections = { ...msg.selected_tokens };
    modifiedSelections["15"] = 3333; // Override one position

    ws.send(JSON.stringify({
      type: 'apply_selection',
      selections: modifiedSelections
    }));
  }
};
```

### Example 4: Error Handling

```javascript
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'error') {
    console.error('Server error:', msg.message);

    // Handle specific errors
    if (msg.message.includes('Model not loaded')) {
      // Load model
      ws.send(JSON.stringify({
        type: 'load_model',
        model_path: 'GSAI-ML/LLaDA-8B-Base'
      }));
    }
  }
};
```

---

## ⚡ Performance Tips

### 1. **Batch Operations**
- Use `select_tokens_only` for quick token selection without forward pass
- Cache `raw_logits` for multiple probability reprocessing

### 2. **Optimize Parameters**
- Lower `top_k` values for faster candidate extraction
- Reduce `gen_length` for faster initialization
- Use appropriate `block_length` for your use case

### 3. **Connection Management**
- Handle WebSocket disconnections gracefully
- Implement reconnection logic with exponential backoff
- Monitor connection state

### 4. **Memory Management**
- Don't store large `raw_logits` arrays unnecessarily
- Process results incrementally for long generations
- Clear unused state data

---

## 🔒 Rate Limits and Constraints

### Message Limits
- Maximum message size: 10MB (for large `raw_logits` arrays)
- No explicit rate limiting (depends on model inference speed)

### Parameter Constraints
- `gen_length`: 1-1024 tokens
- `block_length`: 1-256 tokens
- `top_k`: 1-50000 candidates
- `top_p`: 0.01-1.0
- `softmax_temperature`: 0.1-10.0
- `gumbel_temperature`: 0.0-2.0

### Resource Limits
- Maximum concurrent WebSocket connections: 10
- Model loading timeout: 300 seconds
- Generation step timeout: 60 seconds per step

This API provides complete control over the diffusion language model generation process with real-time feedback and interactive capabilities.