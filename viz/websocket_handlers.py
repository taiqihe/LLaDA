import json
import traceback
from typing import Dict, Any
from dataclasses import asdict
from fastapi import WebSocket

from config import MessageTypes
from models import GenerationParams
from diffusion_model import DiffusionModel
from generation_engine import GenerationEngine


class WebSocketMessageHandler:
    """Handles WebSocket message routing and responses."""

    def __init__(self, diffusion_model: DiffusionModel, generation_engine: GenerationEngine):
        self.diffusion_model = diffusion_model
        self.generation_engine = generation_engine

    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Route messages to appropriate handlers"""
        message_type = message.get("type")

        try:
            if message_type == MessageTypes.LOAD_MODEL:
                await self._handle_load_model(websocket, message)
            elif message_type == MessageTypes.INITIALIZE:
                await self._handle_initialize(websocket, message)
            elif message_type == MessageTypes.STEP:
                await self._handle_step(websocket, message)
            elif message_type == MessageTypes.TOKENIZE_AND_FORWARD:
                await self._handle_tokenize_and_forward(websocket, message)
            elif message_type == MessageTypes.FORWARD_PASS:
                await self._handle_forward_pass(websocket, message)
            elif message_type == MessageTypes.SELECT_TOKENS_ONLY:
                await self._handle_select_tokens_only(websocket, message)
            elif message_type == MessageTypes.REPROCESS_PROBABILITIES:
                await self._handle_reprocess_probabilities(websocket, message)
            elif message_type == MessageTypes.REWIND:
                await self._handle_rewind(websocket, message)
            else:
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
            await websocket.send_text(json.dumps({
                "type": MessageTypes.MODEL_LOAD_RESULT,
                "success": success,
                "model_path": model_path if success else None,
                "error": None if success else f"Failed to load model from {model_path}"
            }))
        except Exception as e:
            print(f"Error in load_model handler: {e}")
            traceback.print_exc()
            await websocket.send_text(json.dumps({
                "type": MessageTypes.MODEL_LOAD_RESULT,
                "success": False,
                "model_path": None,
                "error": f"Error loading model: {str(e)}"
            }))

    async def _handle_initialize(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle generation initialization"""
        prompt = message["prompt"]
        gen_length = message.get("gen_length", 128)
        block_length = message.get("block_length", 32)

        # Tokenize prompt
        prompt_tokens = self.diffusion_model.tokenize_prompt(prompt)

        # Initialize generation
        state = self.generation_engine.initialize_generation(
            prompt_tokens, gen_length, block_length, self.diffusion_model.tokenizer
        )

        await websocket.send_text(json.dumps({
            "type": MessageTypes.STATE_UPDATE,
            "state": asdict(state),
            "is_complete": False
        }))

    async def _handle_step(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle generation step"""
        # Extract parameters
        params = GenerationParams(
            tokens_to_select=message.get("tokens_to_select", 1),
            block_length=message.get("block_length", 32),
            gumbel_temperature=message.get("gumbel_temperature", 0.0),
            remasking_strategy=message.get("remasking", "low_confidence"),
            top_k=message.get("top_k", 10),
            manual_selections=self._convert_manual_selections(message.get("manual_selections", {}))
        )

        # Step generation
        state, is_complete = self.generation_engine.step_generation(
            self.diffusion_model.run_forward_pass,
            self.diffusion_model.tokenizer,
            params
        )

        await websocket.send_text(json.dumps({
            "type": MessageTypes.STATE_UPDATE,
            "state": asdict(state),
            "is_complete": is_complete
        }))

    async def _handle_tokenize_and_forward(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle prompt tokenization and forward pass"""
        prompt = message["prompt"]
        top_k = message.get("top_k", 10)
        gen_length = message.get("gen_length", 128)

        # Tokenize and run forward pass
        prompt_tokens = self.diffusion_model.tokenize_prompt(prompt)
        result = self.diffusion_model.forward_pass_with_prompt(prompt_tokens, gen_length, top_k)

        await websocket.send_text(json.dumps({
            "type": MessageTypes.FORWARD_PASS_RESULT,
            "result": result
        }))

    async def _handle_forward_pass(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle forward pass with provided tokens"""
        tokens = message["tokens"]
        top_k = message.get("top_k", 10)

        result = self.diffusion_model.forward_pass_with_prompt(tokens, top_k)
        await websocket.send_text(json.dumps({
            "type": MessageTypes.FORWARD_PASS_RESULT,
            "result": result
        }))

    async def _handle_select_tokens_only(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle token selection without forward pass"""
        tokens_to_select = message.get("tokens_to_select", 1)
        remasking = message.get("remasking", "low_confidence")
        manual_selections = self._convert_manual_selections(message.get("manual_selections", {}))

        state, is_complete = self.generation_engine.select_tokens_only(
            self.diffusion_model.tokenizer,
            tokens_to_select,
            remasking,
            manual_selections
        )

        await websocket.send_text(json.dumps({
            "type": MessageTypes.STATE_UPDATE,
            "state": asdict(state),
            "is_complete": is_complete
        }))

    async def _handle_reprocess_probabilities(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle probability reprocessing with size limits"""
        print("Received reprocess_probabilities request")
        raw_logits = message.get("raw_logits")
        settings = message.get("settings", {})

        if raw_logits is None:
            raise ValueError("raw_logits is required")

        try:
            # Check input size and truncate if necessary
            if isinstance(raw_logits, list) and len(raw_logits) > 10:
                print(f"Truncating raw_logits from {len(raw_logits)} to 10 positions to avoid large messages")
                raw_logits = raw_logits[:10]

            print(f"Processing {len(raw_logits) if hasattr(raw_logits, '__len__') else 'N/A'} positions")

            result = self.diffusion_model.reprocess_probabilities_with_settings(raw_logits, settings)
            print("Reprocessing completed successfully")

            response = {
                "type": MessageTypes.REPROCESSED_PROBABILITIES_RESULT,
                "result": result
            }

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
            await websocket.send_text(json.dumps({
                "type": MessageTypes.STATE_UPDATE,
                "state": asdict(state),
                "is_complete": False
            }))

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to client"""
        try:
            await websocket.send_text(json.dumps({
                "type": MessageTypes.ERROR,
                "message": error_message
            }))
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
            await websocket.send_text(json.dumps({
                "type": MessageTypes.MODEL_STATUS,
                "is_loaded": True,
                "model_path": self.diffusion_model.model_path
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": MessageTypes.MODEL_STATUS,
                "is_loaded": False,
                "model_path": None
            }))