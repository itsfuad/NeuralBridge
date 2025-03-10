"""
Model Manager Module

This module handles loading, saving, and managing AI models.
It provides functionality to:
- Load models from disk
- Download models from various providers
- Save models for later reuse
- List available models
"""

import os
import json
import pickle
import importlib
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI models, including loading, saving, and configuration."""
    
    def __init__(self, models_dir: str = "models", config_file: str = "models/config.json"):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store model files
            config_file: Path to the model configuration file
        """
        self.models_dir = models_dir
        self.config_file = config_file
        self.loaded_models = {}
        self.model_configs = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Load model configurations if config file exists
        if os.path.exists(config_file):
            self._load_model_configs()
        else:
            # Create default config if it doesn't exist
            self.model_configs = {}
            self._save_model_configs()
    
    def _load_model_configs(self) -> None:
        """Load model configurations from the config file."""
        try:
            with open(self.config_file, 'r') as f:
                self.model_configs = json.load(f)
                logger.info(f"Loaded configurations for {len(self.model_configs)} models")
        except Exception as e:
            logger.error(f"Error loading model configurations: {e}")
            self.model_configs = {}
    
    def _save_model_configs(self) -> None:
        """Save current model configurations to the config file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.model_configs, f, indent=2)
            logger.info(f"Saved configurations for {len(self.model_configs)} models")
        except Exception as e:
            logger.error(f"Error saving model configurations: {e}")
    
    def list_available_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of model names
        """
        # Models explicitly in our config
        models = list(self.model_configs.keys())
        
        # Add any model files in the models directory that might not be in config
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl') or filename.endswith('.bin'):
                model_name = os.path.splitext(filename)[0]
                if model_name not in models:
                    models.append(model_name)
        
        return models
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a model by name.
        
        If the model is already loaded, return the cached instance.
        If not, try to load from disk or download if necessary.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model instance
        """
        # Return cached model if already loaded
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        # Check if model exists in configuration
        if model_name in self.model_configs:
            return self._load_configured_model(model_name)
        
        # Try to load from disk based on filename
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            return self._load_model_from_file(model_path, model_name)
        
        # If we can't find the model, raise an error
        raise ValueError(f"Model '{model_name}' not found and no configuration exists to download it.")
    
    def _load_configured_model(self, model_name: str) -> Any:
        """Load a model based on its configuration."""
        config = self.model_configs[model_name]
        model_type = config.get('type', 'custom')
        
        if model_type == 'huggingface':
            return self._load_from_huggingface(model_name, config)
        elif model_type == 'openai':
            return self._load_openai_model(model_name, config)
        elif model_type == 'custom':
            return self._load_custom_model(model_name, config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_model_from_file(self, file_path: str, model_name: str) -> Any:
        """Load a model from a file."""
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from file: {file_path}")
            
            # Cache the loaded model
            self.loaded_models[model_name] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}")
            raise ValueError(f"Failed to load model from {file_path}: {str(e)}")
    
    def _load_from_huggingface(self, model_name: str, config: Dict) -> Any:
        """Load a model from Hugging Face."""
        try:
            # Import here to avoid dependency if not needed
            from transformers import AutoModel, AutoTokenizer
            
            model_id = config.get('model_id', model_name)
            cache_dir = os.path.join(self.models_dir, model_name)
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            
            # Create a wrapper object containing both model and tokenizer
            model_obj = {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
            
            # Cache the loaded model
            self.loaded_models[model_name] = model_obj
            logger.info(f"Loaded Hugging Face model: {model_id}")
            return model_obj
            
        except ImportError:
            logger.error("Failed to import transformers. Please install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading Hugging Face model {model_name}: {e}")
            raise ValueError(f"Failed to load Hugging Face model: {str(e)}")
    
    def _load_openai_model(self, model_name: str, config: Dict) -> Any:
        """Load an OpenAI model."""
        try:
            # Import here to avoid dependency if not needed
            import openai
            
            # OpenAI models don't need to be downloaded, just create a wrapper
            # with the configuration
            openai.api_key = config.get('api_key', os.environ.get('OPENAI_API_KEY'))
            
            if not openai.api_key:
                raise ValueError("OpenAI API key not found. Set it in the config or as OPENAI_API_KEY environment variable.")
            
            model_obj = {
                'type': 'openai',
                'model_id': config.get('model_id', 'gpt-3.5-turbo'),
                'config': config
            }
            
            # Cache the model configuration
            self.loaded_models[model_name] = model_obj
            logger.info(f"Configured OpenAI model: {model_obj['model_id']}")
            return model_obj
            
        except ImportError:
            logger.error("Failed to import openai. Please install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error configuring OpenAI model {model_name}: {e}")
            raise ValueError(f"Failed to configure OpenAI model: {str(e)}")
    
    def _load_custom_model(self, model_name: str, config: Dict) -> Any:
        """Load a custom model using the specified module and class."""
        try:
            module_name = config.get('module')
            class_name = config.get('class')
            
            if not module_name or not class_name:
                raise ValueError("Custom model configuration must include 'module' and 'class'")
            
            # Dynamically import the module and get the class
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            
            # Get model init parameters
            params = config.get('params', {})
            
            # Initialize the model
            model = model_class(**params)
            
            # Cache the loaded model
            self.loaded_models[model_name] = model
            logger.info(f"Loaded custom model: {module_name}.{class_name}")
            return model
            
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading custom model {model_name}: {e}")
            raise ValueError(f"Failed to load custom model: {str(e)}")
    
    def save_model(self, model: Any, model_name: str, config: Optional[Dict] = None) -> str:
        """
        Save a model to disk.
        
        Args:
            model: The model to save
            model_name: Name to save the model as
            config: Optional configuration for the model
            
        Returns:
            Path to the saved model file
        """
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Determine the save path
            save_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            
            # Save the model
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Update configuration if provided
            if config:
                self.model_configs[model_name] = config
                self._save_model_configs()
            
            logger.info(f"Saved model to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise ValueError(f"Failed to save model: {str(e)}")
    
    def register_model(self, model_name: str, config: Dict) -> None:
        """
        Register a model configuration without loading it.
        
        Args:
            model_name: Name to register the model as
            config: Configuration for the model
        """
        self.model_configs[model_name] = config
        self._save_model_configs()
        logger.info(f"Registered model configuration: {model_name}")
    
    def unregister_model(self, model_name: str) -> bool:
        """
        Unregister a model configuration and optionally delete its files.
        
        Args:
            model_name: Name of the model to unregister
            
        Returns:
            True if the model was unregistered, False otherwise
        """
        # Remove from loaded models if present
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
        
        # Remove from configurations if present
        if model_name in self.model_configs:
            del self.model_configs[model_name]
            self._save_model_configs()
            logger.info(f"Unregistered model: {model_name}")
            return True
        
        return False 