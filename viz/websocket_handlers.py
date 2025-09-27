import json
import traceback
from typing import Dict, Any
from dataclasses import asdict
from fastapi import WebSocket

from models import MessageTypes
from diffusion_model import DiffusionModel
from token_tracker import TokenTracker
from logger_config import websocket_logger


class WebSocketMessageHandler:
    """Handles WebSocket message routing and responses."""

    def __init__(self, diffusion_model: DiffusionModel, token_tracker: TokenTracker, config: Dict):
        self.diffusion_model = diffusion_model
        self.token_tracker = token_tracker
        self.config = config

    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Route messages to appropriate handlers"""
        message_type = message.get("type")
        websocket_logger.info(f"Received message: {message_type}")

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
                    websocket_logger.warning(f"Unknown message type: {message_type}")
                    await self._send_error(websocket, f"Unknown message type: {message_type}")

        except Exception as e:
            websocket_logger.error(f"Error handling message type {message_type}: {e}")
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
        try:
            # Validate required parameters
            prompt = message.get("prompt")
            if not prompt or not isinstance(prompt, str):
                await self._send_error(websocket, "Invalid prompt: must be a non-empty string.")
                return

            gen_length = message.get("gen_length", self.config["gen_length"])
            if not isinstance(gen_length, int) or gen_length < 1 or gen_length > 1024:
                await self._send_error(websocket, "Invalid gen_length: must be an integer between 1 and 1024.")
                return

            block_length = message.get("block_length", self.config["block_length"])
            if not isinstance(block_length, int) or block_length < 1 or block_length > gen_length:
                await self._send_error(websocket, "Invalid block_length: must be an integer between 1 and gen_length.")
                return

            # Check if model is loaded
            if not self.diffusion_model.is_model_loaded():
                await self._send_error(websocket, "No model loaded. Please load a model first.")
                return

            # Tokenize prompt
            prompt_tokens = self.diffusion_model.tokenize_prompt(prompt)

            # Initialize generation
            state = self.token_tracker.initialize_generation(
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
        except Exception as e:
            await self._send_error(websocket, f"Error initializing generation: {str(e)}")

    async def _handle_auto_select(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle automatic token selection - placeholder implementation"""
        try:
            # Get current generation state
            if not self.token_tracker.current_state:
                await self._send_error(websocket, "No generation initialized. Please initialize first.")
                return

            # Check if we have cached forward pass results
            if not self.diffusion_model.has_cached_results():
                await self._send_error(websocket, "No forward pass results available. Please run forward pass first.")
                return

            # Validate required parameters
            tokens_to_select = message.get("tokens_to_select", self.config["tokens_to_select"])
            if not isinstance(tokens_to_select, int) or tokens_to_select < 1:
                await self._send_error(websocket, "Invalid tokens_to_select: must be a positive integer.")
                return

            strategy = message.get("strategy", self.config["remasking"])
            selection_method = message.get("selection", self.config["selection"])

            # Get cached results
            cached_results = self.diffusion_model.get_cached_results()

            # Perform auto selection using cached data
            selected_tokens = self.token_tracker.auto_select(
                probabilities=cached_results['probs'],
                x0=cached_results['x0'],
                strategy=strategy,
                max_tokens=tokens_to_select,
                selection_method=selection_method
            )

            await websocket.send_text(
                json.dumps(
                    {
                        "type": MessageTypes.STATE_UPDATE,
                        "selected_tokens": selected_tokens,
                        "message": f"Auto-selected {len(selected_tokens)} tokens using {strategy} strategy.",
                    }
                )
            )
        except Exception as e:
            await self._send_error(websocket, f"Error in auto selection: {str(e)}")

    async def _handle_apply_selection(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle applying token selections to the generation state"""
        try:
            # Get current generation state
            if not self.token_tracker.current_state:
                await self._send_error(websocket, "No generation initialized. Please initialize first.")
                return

            # Get selections from message
            selections = message.get("selections", {})
            if not selections or not isinstance(selections, dict):
                await self._send_error(websocket, "No valid selections provided. Must be a dictionary of position: token_id.")
                return

            # Validate selections format
            try:
                validated_selections = {}
                for pos, token_id in selections.items():
                    pos_int = int(pos)
                    token_id_int = int(token_id)
                    if pos_int < 0 or token_id_int < 0:
                        await self._send_error(websocket, f"Invalid selection: position {pos_int} and token_id {token_id_int} must be non-negative.")
                        return
                    validated_selections[pos_int] = token_id_int
            except (ValueError, TypeError):
                await self._send_error(websocket, "Invalid selections format. Positions and token IDs must be integers.")
                return

            # Apply selections
            updated_state = self.token_tracker.apply_selections(validated_selections, self.diffusion_model.tokenizer)

            await websocket.send_text(
                json.dumps(
                    {
                        "type": MessageTypes.STATE_UPDATE,
                        "state": asdict(updated_state),
                    }
                )
            )
        except Exception as e:
            await self._send_error(websocket, f"Error applying selections: {str(e)}")

    async def _handle_forward_pass(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle forward pass with provided tokens"""
        try:
            # Validate tokens
            tokens = message.get("tokens")
            if not tokens or not isinstance(tokens, list):
                await self._send_error(websocket, "Invalid tokens: must be a non-empty list of token IDs.")
                return

            # Validate token IDs
            try:
                validated_tokens = [int(t) for t in tokens]
                if any(t < 0 for t in validated_tokens):
                    await self._send_error(websocket, "Invalid tokens: all token IDs must be non-negative integers.")
                    return
            except (ValueError, TypeError):
                await self._send_error(websocket, "Invalid tokens: all items must be integers.")
                return

            top_k = message.get("top_k", 10)
            if not isinstance(top_k, int) or top_k < 1 or top_k > 50000:
                await self._send_error(websocket, "Invalid top_k: must be an integer between 1 and 50000.")
                return

            # Check if model is loaded
            if not self.diffusion_model.is_model_loaded():
                await self._send_error(websocket, "No model loaded. Please load a model first.")
                return

            visual_top_k = message.get("visual_top_k", self.config["visual_top_k"])
            actual_top_k = message.get("actual_top_k", self.config["actual_top_k"])
            top_p = message.get("top_p", self.config["top_p"])

            result = self.diffusion_model.forward_pass_with_prompt(
                validated_tokens,
                gen_length=len(validated_tokens),
                visual_top_k=visual_top_k,
                actual_top_k=actual_top_k,
                top_p=top_p
            )
            await websocket.send_text(json.dumps({"type": MessageTypes.FORWARD_PASS_RESULT, "result": result}))
        except Exception as e:
            await self._send_error(websocket, f"Error in forward pass: {str(e)}")

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
        try:
            # Validate step parameter
            step = message.get("step")
            if step is None or not isinstance(step, int) or step < 0:
                await self._send_error(websocket, "Invalid step: must be a non-negative integer.")
                return

            # Check if generation is initialized
            if not self.token_tracker.current_state:
                await self._send_error(websocket, "No generation initialized. Please initialize first.")
                return

            state = self.token_tracker.rewind_to_step(step)
            if state:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": MessageTypes.STATE_UPDATE,
                            "state": asdict(state),
                        }
                    )
                )
            else:
                await self._send_error(websocket, f"Cannot rewind to step {step}. Step not found in cache.")
        except Exception as e:
            await self._send_error(websocket, f"Error rewinding: {str(e)}")

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
