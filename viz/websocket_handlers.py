import json
import traceback
from typing import Dict, Any
from dataclasses import asdict
from fastapi import WebSocket

from models import MessageTypes, GenerationParams
from diffusion_model import DiffusionModel
from token_tracker import TokenTracker


class WebSocketMessageHandler:
    """Handles WebSocket message routing and responses."""

    def __init__(self, diffusion_model: DiffusionModel, token_tracker: TokenTracker, config: Dict):
        self.diffusion_model = diffusion_model
        self.token_tracker = token_tracker
        self.config = config

    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Route messages to appropriate handlers"""
        message_type = message.get("type")

        try:
            match message_type:
                case MessageTypes.LOAD_MODEL:
                    await self._handle_load_model(websocket, message)
                case MessageTypes.INITIALIZE:
                    await self._handle_initialize(websocket, message)
                case MessageTypes.AUTO_SELECT:
                    await self._handle_auto_select(websocket, message)
                case MessageTypes.FORWARD_PASS:
                    await self._handle_forward_pass(websocket, message)
                case MessageTypes.APPLY_SELECTION:
                    await self._handle_apply_selection(websocket, message)
                case MessageTypes.REPROCESS_PROBABILITIES:
                    await self._handle_reprocess_probabilities(websocket, message)
                case MessageTypes.REWIND:
                    await self._handle_rewind(websocket, message)
                case _:
                    await self._send_error(websocket, f"Unknown message type: {message_type}")

        except Exception as e:
            print(f"Error handling message type {message_type}: {e}")
            traceback.print_exc()
            await self._send_error(websocket, f"Error processing {message_type}: {str(e)}")

    async def _handle_load_model(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle model loading"""
        try:
            model_path = message["model_path"]
            success = self.diffusion_model.load_model(model_path)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": MessageTypes.MODEL_LOAD_RESULT,
                        "success": success,
                        "model_path": model_path if success else None,
                        "error": None if success else f"Failed to load model from {model_path}",
                    }
                )
            )
        except Exception as e:
            print(f"Error in load_model handler: {e}")
            traceback.print_exc()
            await websocket.send_text(
                json.dumps(
                    {
                        "type": MessageTypes.MODEL_LOAD_RESULT,
                        "success": False,
                        "model_path": None,
                        "error": f"Error loading model: {str(e)}",
                    }
                )
            )

    async def _handle_initialize(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle generation initialization"""
        prompt = message["prompt"]
        gen_length = message.get("gen_length", self.config["gen_length"])
        block_length = message.get("block_length", self.config["block_length"])

        # Tokenize prompt
        prompt_tokens = self.diffusion_model.tokenize_prompt(prompt)

        # Initialize generation
        state = self.generation_engine.initialize_generation(
            prompt_tokens, gen_length, block_length, self.diffusion_model.tokenizer
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": MessageTypes.STATE_UPDATE,
                    "state": asdict(state),
                }
            )
        )

    async def _handle_auto_select(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle generation step"""
        probs = self.diffusion_model.get_probabilities()
        params = {
            "tokens_to_select": message.get("tokens_to_select", self.config["tokens_to_select"]),
            "block_length": message.get("block_length", self.config["block_length"]),
            "strategy": message.get("remasking", self.config["remasking"]),
            "selection": message.get("selection", self.config["selection"]),
        }

        tokens = self.token_tracker.auto_select(probs, **params)
        await websocket.send_text(
            json.dumps(
                {
                    "type": MessageTypes.TOKEN_SELECTIONS,
                    "tokens": tokens,
                }
            )
        )

    async def _handle_apply_selection(self, websocket: WebSocket, message: Dict[str, Any]):
        pass

    async def _handle_forward_pass(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle forward pass with provided tokens"""
        tokens = message["tokens"]
        top_k = message.get("top_k", 10)

        result = self.diffusion_model.forward_pass_with_prompt(tokens, top_k)
        await websocket.send_text(json.dumps({"type": MessageTypes.FORWARD_PASS_RESULT, "result": result}))

    async def _handle_reprocess_probabilities(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle probability reprocessing with size limits"""
        print("Received reprocess_probabilities request")
        raw_logits = message.get("raw_logits")
        settings = message.get("settings", {})

        if raw_logits is None:
            raise ValueError("raw_logits is required")

        try:
            print(f"Processing {len(raw_logits) if hasattr(raw_logits, '__len__') else 'N/A'} positions")

            result = self.diffusion_model.reprocess_probabilities_with_settings(raw_logits, settings)
            print("Reprocessing completed successfully")

            response = {"type": MessageTypes.REPROCESSED_PROBABILITIES_RESULT, "result": result}

            response_json = json.dumps(response)
            print(f"Response size: {len(response_json)} characters")

            await websocket.send_text(response_json)
            print("Response sent to client")

        except Exception as e:
            print(f"Error in reprocessing: {e}")
            raise

    async def _handle_rewind(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle generation rewind"""
        step = message["step"]
        state = self.generation_engine.rewind_to_step(step)
        if state:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": MessageTypes.STATE_UPDATE,
                        "state": asdict(state),
                    }
                )
            )

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to client"""
        try:
            await websocket.send_text(json.dumps({"type": MessageTypes.ERROR, "message": error_message}))
        except Exception as send_error:
            print(f"Failed to send error message: {send_error}")

    def _convert_manual_selections(self, manual_selections: Dict[str, Any]) -> Dict[int, int]:
        """Convert string keys to int for manual selections"""
        if not manual_selections:
            return {}
        return {int(k): v for k, v in manual_selections.items()}

    async def send_model_status(self, websocket: WebSocket):
        """Send current model status to client"""
        if self.diffusion_model.is_model_loaded():
            await websocket.send_text(
                json.dumps(
                    {
                        "type": MessageTypes.MODEL_STATUS,
                        "is_loaded": True,
                        "model_path": self.diffusion_model.model_path,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps({"type": MessageTypes.MODEL_STATUS, "is_loaded": False, "model_path": None})
            )
