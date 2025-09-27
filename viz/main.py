import json
import sys
import os
import argparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_HOST, DEFAULT_PORT
from diffusion_model import DiffusionModel
from generation_engine import GenerationEngine
from websocket_handlers import WebSocketMessageHandler


app = FastAPI()


class DiffusionVisualizer:
    """Main visualizer class that coordinates all components."""

    def __init__(self, model_path: str = None):
        self.diffusion_model = DiffusionModel()
        self.generation_engine = GenerationEngine()
        self.message_handler = WebSocketMessageHandler(self.diffusion_model, self.generation_engine)

        if model_path:
            self.diffusion_model.load_model(model_path)

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.diffusion_model.is_model_loaded()

    @property
    def model_path(self) -> str:
        """Get current model path"""
        return self.diffusion_model.model_path


# Global visualizer instance
visualizer = None


def initialize_visualizer(model_path: str = None):
    """Initialize the global visualizer instance"""
    global visualizer
    visualizer = DiffusionVisualizer(model_path)


@app.get("/")
async def get_visualizer():
    """Serve the main HTML interface"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()

    # Send current model status when client connects
    await visualizer.message_handler.send_model_status(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await visualizer.message_handler.handle_message(websocket, message)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error in WebSocket connection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Diffusion Language Model Visualizer")
    parser.add_argument("--model-path", "-m", type=str, help="Path to the model (local path or HuggingFace model ID)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")

    args = parser.parse_args()

    # Initialize visualizer with optional model path
    initialize_visualizer(args.model_path)

    print(f"Starting Diffusion Language Model Visualizer")
    print(f"Server: http://{args.host}:{args.port}")
    if args.model_path:
        print(f"Model: {args.model_path}")
    else:
        print("Model: Load via web interface")

    uvicorn.run(app, host=args.host, port=args.port)