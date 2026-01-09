# LLaDA Diffusion Language Model Visualizer

A web-based interactive visualizer for exploring diffusion language models, featuring real-time generation visualization, probability inspection, and step-by-step diffusion process analysis.

![LLaDA Visualizer Demo](https://img.shields.io/badge/Status-Live-green.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)

## 🎯 Features

- **Real-time Generation Visualization**: Watch diffusion models generate text with live probability updates
- **Interactive Token Selection**: Manually select tokens during the generation process or use auto-selection strategies
- **Probability Analysis**: Inspect token probabilities with customizable temperature and sampling settings
- **Dual Token Restrictions**: Separate visual and actual candidate limits for flexible exploration
- **Forward Pass with Loading Indicator**: Visual feedback during model inference
- **Remasking Strategies**: Experiment with different token selection strategies (low/high confidence, entropy-based, random)
- **Probability Reprocessing**: Dynamically adjust temperature and sampling parameters without re-running inference
- **WebSocket API**: Real-time communication for responsive visualization
- **Modular Architecture**: Clean, maintainable codebase with clear separation of concerns

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Visualizer

#### Option 1: With Local Model
```bash
python main.py --model-path /path/to/your/model
```

#### Option 2: Load Model via Web Interface
```bash
python main.py
```
Then navigate to `http://localhost:8000` and load your model through the web interface.

#### Option 3: Custom Host/Port
```bash
python main.py --host 0.0.0.0 --port 8080 --model-path GSAI-ML/LLaDA-8B-Base
```

### Using with HuggingFace Models

The visualizer supports both local models and HuggingFace model IDs:

```bash
# Use HuggingFace model ID
python main.py --model-path GSAI-ML/LLaDA-8B-Base

# Use local model path
python main.py --model-path /Users/user/models/LLaDA-8B-Base
```

## 📋 Usage

### Basic Generation Workflow

1. **Load Model**: Load a diffusion model from HuggingFace or local path
2. **Initialize Generation**: Enter a prompt and configure generation length
3. **Run Forward Pass**: Execute model inference to get token probabilities (shows loading indicator)
4. **Inspect & Adjust Probabilities**: View candidates and optionally adjust temperature/sampling settings
5. **Select Tokens**: Use auto-selection strategies or manually click tokens
6. **Apply Selections**: Apply selected tokens to update the generation state
7. **Repeat**: Continue with steps 3-6 until generation is complete

### Advanced Features

#### Auto-Selection Strategies
- **Low Confidence**: Select tokens with lowest prediction confidence
- **High Confidence**: Select tokens with highest prediction confidence
- **Low Entropy**: Select positions with most certain probability distributions
- **High Entropy**: Select positions with most uncertain probability distributions
- **Random**: Random token selection

#### Probability Processing Controls
- **Softmax Temperature**: Controls prediction sharpness (lower = more focused)
- **Gumbel Noise**: Adds stochastic noise for more diverse sampling
- **Visual Top-K**: Number of candidates shown in UI (for exploration)
- **Actual Top-K**: Number of candidates available for selection (for control)
- **Top-P (Nucleus)**: Cumulative probability threshold for filtering

#### Manual Token Selection
- Click any candidate token to select it for a position
- Combine auto-selection with manual overrides
- Clear all selections or apply them to the generation state

## 📁 Project Structure

```
viz/
├── main.py                    # Server entry point and coordination
├── models.py                  # Data classes and type definitions
├── config.py                  # Configuration constants
├── diffusion_model.py         # Core model operations and loading
├── generation_engine.py       # Generation state management
├── probability_processor.py   # Probability calculations and filtering
├── websocket_handlers.py      # WebSocket message routing
├── index.html                 # Web interface
└── requirements.txt           # Python dependencies
```

## 🏗️ Architecture

The visualizer follows a modular architecture with clear separation of concerns:

- **DiffusionModel**: Handles model loading and forward passes
- **GenerationEngine**: Manages generation state and token selection
- **ProbabilityProcessor**: Processes logits and applies filtering
- **WebSocketMessageHandler**: Routes and handles client messages
- **DiffusionVisualizer**: Coordinates all components

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 🔌 API Reference

The visualizer exposes a WebSocket API for real-time interaction:

### Message Types
- `load_model` - Load a diffusion model
- `initialize` - Initialize generation with prompt
- `forward_pass` - Run model inference on current state
- `auto_select` - Automatically select tokens using ranking strategies
- `apply_selection` - Apply selected tokens to generation state
- `reprocess_probabilities` - Reprocess probabilities with new settings
- `rewind` - Rewind to previous generation step

See [API.md](API.md) for complete API documentation.

## 🛠️ Configuration

### Model Settings
- `device`: CUDA/CPU device selection (auto-detected)
- `mask_id`: Token ID used for masking (default: 126336)
- `dtype`: Model precision (default: auto)

### Generation Parameters
- `gen_length`: Number of tokens to generate (default: 128, range: 1-512)
- `visual_top_k`: Candidates shown in UI (default: 20, range: 1-100)
- `actual_top_k`: Candidates for selection (default: 10, range: 1-100)
- `top_p`: Nucleus sampling threshold (default: 1.0, range: 0.01-1.0)
- `softmax_temperature`: Prediction sharpness (default: 1.0, range: 0.1-5.0)
- `gumbel_temperature`: Noise temperature (default: 0.0, range: 0.0-2.0)

### Server Settings
- `host`: Server host (default: 0.0.0.0)
- `port`: Server port (default: 8000)

## 🐛 Troubleshooting

### Common Issues

**Model Loading Fails**
```bash
Error loading model: ...
```
- Verify model path exists
- Check model format compatibility
- Ensure sufficient RAM/VRAM

**WebSocket Connection Issues**
```bash
WebSocket connection failed
```
- Check firewall settings
- Verify port availability
- Try different host/port combination

**Generation Stalls**
```bash
Generation not progressing
```
- Check for NaN/Inf in probabilities
- Verify mask token ID is correct
- Review generation parameters

### Performance Tips

- Use CUDA if available for faster inference
- Reduce `gen_length` for faster initialization
- Lower `top_k` values for faster probability calculation
- Use `bfloat16` precision to reduce memory usage

## 📚 Examples

### Basic Usage
```python
# Initialize visualizer
visualizer = DiffusionVisualizer("/path/to/model")

# Check model status
if visualizer.is_model_loaded():
    print(f"Model loaded: {visualizer.model_path}")
```

### Custom Generation Parameters
```javascript
// WebSocket message for custom generation
{
  "type": "initialize",
  "prompt": "The future of AI is",
  "gen_length": 64
}

// Run forward pass with custom restrictions
{
  "type": "forward_pass",
  "gen_length": 64,
  "visual_top_k": 30,
  "actual_top_k": 15,
  "top_p": 0.9
}

// Auto-select tokens using a strategy
{
  "type": "auto_select",
  "strategy": "low_confidence",
  "max_tokens": 2
}

// Apply selections
{
  "type": "apply_selection",
  "selections": {
    "10": 1234,
    "11": 5678
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the modular architecture
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the same terms as the main LLaDA project.

## 🙏 Acknowledgments

- Built for the LLaDA (Large Language Diffusion with mAsking) project
- Thanks to the HuggingFace team for model hosting and transformers library
- FastAPI and WebSocket libraries for real-time communication