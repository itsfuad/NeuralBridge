from abc import ABC
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
import json
import logging
import asyncio
from pathlib import Path
import shutil

class ModelInterface(ABC):
    def __init__(self):
        self.config = {}
        self.model_info = {
            "name": "base_model",
            "version": "1.0.0",
            "capabilities": []
        }

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the model."""
        try:
            # Base implementation that can be overridden
            return "This is a base model response"
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return self.model_info

    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the model."""
        try:
            # Base implementation that can be overridden
            response = "This is a streaming response"
            for char in response:
                yield char
                await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error in stream generation: {str(e)}")
            raise

    def get_model_config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        return self.config

    def set_model_config(self, config: Dict[str, Any]) -> None:
        """Update the model configuration."""
        self.config.update(config)

class MemoryInterface(ABC):
    def __init__(self):
        self.messages = []
        self.message_ids = {}
        self.stats = {
            "total_messages": 0,
            "total_size": 0,
            "last_cleared": None
        }

    def add_message(self, message: Dict[str, str]) -> None:
        """Add a message to memory."""
        message_id = str(len(self.messages))
        message["id"] = message_id
        message["timestamp"] = datetime.now()
        self.messages.append(message)
        self.message_ids[message_id] = message
        self.stats["total_messages"] += 1
        self.stats["total_size"] += len(json.dumps(message))

    def get_relevant_context(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Get relevant context for a query."""
        # Simple implementation - return most recent messages
        return self.messages[-limit:]

    def clear(self) -> None:
        """Clear the memory."""
        self.messages.clear()
        self.message_ids.clear()
        self.stats["last_cleared"] = datetime.now()
        self.stats["total_messages"] = 0
        self.stats["total_size"] = 0

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.stats

    def search_memory(self, query: str, limit: int = 5, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Search memory with time constraints."""
        filtered_messages = self.messages
        if start_time:
            filtered_messages = [m for m in filtered_messages 
                               if m["timestamp"] >= start_time]
        if end_time:
            filtered_messages = [m for m in filtered_messages 
                               if m["timestamp"] <= end_time]
        return filtered_messages[-limit:]

    def get_memory_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific message by ID."""
        return self.message_ids.get(message_id)

class DatabaseInterface(ABC):
    def __init__(self, db_path: str = "data/db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "total_items": 0,
            "total_size": 0,
            "last_backup": None
        }

    def store(self, key: str, value: Any) -> None:
        """Store a value in the database."""
        file_path = self.db_path / f"{key}.json"
        with open(file_path, 'w') as f:
            json.dump(value, f)
        self.stats["total_items"] += 1
        self.stats["total_size"] += len(json.dumps(value))

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from the database."""
        file_path = self.db_path / f"{key}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar items in the database."""
        results = []
        for file_path in self.db_path.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if query.lower() in str(data).lower():
                    results.append(data)
                    if len(results) >= limit:
                        break
        return results

    def delete(self, key: str) -> bool:
        """Delete a value from the database."""
        file_path = self.db_path / f"{key}.json"
        if file_path.exists():
            file_path.unlink()
            self.stats["total_items"] -= 1
            return True
        return False

    def update(self, key: str, value: Any) -> bool:
        """Update a value in the database."""
        if self.retrieve(key) is not None:
            self.store(key, value)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.stats

    def backup(self, path: str) -> bool:
        """Create a backup of the database."""
        try:
            backup_path = Path(path)
            backup_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.db_path, backup_path / "db", dirs_exist_ok=True)
            self.stats["last_backup"] = datetime.now()
            return True
        except Exception as e:
            logging.error(f"Backup failed: {str(e)}")
            return False

    def restore(self, path: str) -> bool:
        """Restore the database from a backup."""
        try:
            backup_path = Path(path) / "db"
            if backup_path.exists():
                shutil.rmtree(self.db_path)
                shutil.copytree(backup_path, self.db_path)
                return True
            return False
        except Exception as e:
            logging.error(f"Restore failed: {str(e)}")
            return False

class StreamInterface(ABC):
    def __init__(self):
        self.config = {
            "chunk_size": 100,
            "delay": 0.1,
            "max_retries": 3
        }

    async def stream_response(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream the response from the model."""
        try:
            response = "This is a streamed response"
            for i in range(0, len(response), self.config["chunk_size"]):
                chunk = response[i:i + self.config["chunk_size"]]
                yield chunk
                await asyncio.sleep(self.config["delay"])
        except Exception as e:
            await self.handle_stream_error(e)

    async def get_stream_config(self) -> Dict[str, Any]:
        """Get streaming configuration."""
        return self.config

    async def set_stream_config(self, config: Dict[str, Any]) -> None:
        """Update streaming configuration."""
        self.config.update(config)

    async def handle_stream_error(self, error: Exception) -> AsyncGenerator[str, None]:
        """Handle streaming errors."""
        error_message = f"Stream error: {str(error)}"
        yield error_message
        logging.error(error_message)

class MultiModalInterface(ABC):
    def __init__(self):
        self.config = {
            "max_image_size": 1024,
            "max_audio_duration": 30,
            "supported_formats": ["text", "image", "audio"]
        }

    def process_text(self, text: str) -> str:
        """Process text input."""
        return text.strip()

    def process_image(self, image_data: bytes) -> str:
        """Process image input."""
        # Base implementation - return base64 encoded image
        import base64
        return base64.b64encode(image_data).decode('utf-8')

    def process_audio(self, audio_data: bytes) -> str:
        """Process audio input."""
        # Base implementation - return base64 encoded audio
        import base64
        return base64.b64encode(audio_data).decode('utf-8')

    def get_processor_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return self.config

    def set_processor_config(self, config: Dict[str, Any]) -> None:
        """Update processor configuration."""
        self.config.update(config)

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate input data."""
        for key, value in input_data.items():
            if key not in self.config["supported_formats"]:
                return False, f"Unsupported format: {key}"
        return True, None

    def get_supported_modalities(self) -> List[str]:
        """Get list of supported modalities."""
        return self.config["supported_formats"]

class ModelManagerInterface(ABC):
    def __init__(self):
        self.loaded_models = {}
        self.model_status = {}

    def load_model(self, model_name: str, **kwargs) -> ModelInterface:
        """Load a model by name."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Base implementation - create a new model instance
        model = ModelInterface()
        self.loaded_models[model_name] = model
        self.model_status[model_name] = {
            "status": "loaded",
            "load_time": datetime.now(),
            "memory_usage": 0
        }
        return model

    def unload_model(self, model_name: str) -> bool:
        """Unload a model by name."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.model_status[model_name]
            return True
        return False

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self.loaded_models.keys())

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status of a specific model."""
        return self.model_status.get(model_name, {})

    def register_model(self, model_name: str, model_class: type, **kwargs) -> bool:
        """Register a new model class."""
        try:
            self.loaded_models[model_name] = model_class()
            return True
        except Exception as e:
            logging.error(f"Failed to register model {model_name}: {str(e)}")
            return False

class ErrorHandlerInterface(ABC):
    def __init__(self):
        self.error_stats = {
            "total_errors": 0,
            "error_types": {},
            "last_error": None
        }

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an error and return appropriate response."""
        error_type = type(error).__name__
        self.error_stats["total_errors"] += 1
        self.error_stats["error_types"][error_type] = self.error_stats["error_types"].get(error_type, 0) + 1
        self.error_stats["last_error"] = datetime.now()
        
        return {
            "error": str(error),
            "type": error_type,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log an error with context."""
        error_info = self.handle_error(error, context)
        logging.error(f"Error occurred: {json.dumps(error_info)}")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_stats 