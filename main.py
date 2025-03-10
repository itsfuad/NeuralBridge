#!/usr/bin/env python
# NeuralBridge
# A flexible system for running AI models with enhanced memory and database integration

import argparse
import os
import sys
import logging
from src.model_manager import ModelManager
from src.memory_manager import MemoryManager
from src.database import DatabaseManager
from src.chat import ChatSession

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple default model that can be used when no other models are available
class DefaultModel:
    """A very simple model that provides basic responses."""
    
    def generate(self, messages):
        """Generate a simple response based on user input."""
        # Get the last user message
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        
        if user_messages:
            last_message = user_messages[-1]['content'].lower()
            
            # Very basic response patterns
            if "hello" in last_message or "hi" in last_message:
                return "Hello! I'm a very basic default model. You should configure a more advanced model for better responses."
            elif "help" in last_message:
                return "I'm a basic placeholder model. Try configuring a real AI model using the instructions in the README."
            elif "who are you" in last_message or "what are you" in last_message:
                return "I'm the default fallback model for the AI Model Runner System. I have very limited capabilities."
            
        return "I'm a simple placeholder model. Please configure a real AI model for better responses."

# Dictionary of common model configurations
COMMON_MODELS = {
    # OpenAI models
    "gpt-3.5-turbo": {
        "type": "openai",
        "model_id": "gpt-3.5-turbo",
        "config": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    },
    "gpt-4": {
        "type": "openai",
        "model_id": "gpt-4",
        "config": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    },
    
    # Hugging Face models
    "llama2-7b": {
        "type": "huggingface",
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "config": {
            "use_auth_token": True  # Will prompt user for token if needed
        }
    },
    "mistral-7b": {
        "type": "huggingface",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "config": {}
    },
    "phi-2": {
        "type": "huggingface",
        "model_id": "microsoft/phi-2",
        "config": {}
    },
    "gemma-7b": {
        "type": "huggingface",
        "model_id": "google/gemma-7b-it",
        "config": {
            "use_auth_token": True  # Will prompt user for token if needed
        }
    }
}

def register_common_model(model_manager, model_name):
    """
    Register a common model configuration.
    
    Args:
        model_manager: ModelManager instance
        model_name: Name of the model to register
        
    Returns:
        True if registered successfully, False otherwise
    """
    if model_name in COMMON_MODELS:
        config = COMMON_MODELS[model_name]
        
        # For OpenAI models, check for API key
        if config["type"] == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print(f"To use {model_name}, you need to set your OpenAI API key.")
                print("You can do this by setting the OPENAI_API_KEY environment variable.")
                print("For example: export OPENAI_API_KEY=your_api_key (Linux/Mac)")
                print("or: set OPENAI_API_KEY=your_api_key (Windows)")
                return False
                
        # For Hugging Face models that need authentication
        if config["type"] == "huggingface" and config.get("config", {}).get("use_auth_token"):
            token = os.environ.get("HUGGINGFACE_TOKEN")
            if not token:
                print(f"To use {model_name}, you need a Hugging Face token with the appropriate permissions.")
                print("You can do this by setting the HUGGINGFACE_TOKEN environment variable.")
                print("For example: export HUGGINGFACE_TOKEN=your_token (Linux/Mac)")
                print("or: set HUGGINGFACE_TOKEN=your_token (Windows)")
                return False
            
            # Update the config with the token
            config["config"]["use_auth_token"] = token
        
        # Register the model
        model_manager.register_model(model_name, config)
        logger.info(f"Registered common model: {model_name}")
        return True
    
    return False

def create_default_model():
    """
    Create and return a new instance of the default model.
    """
    return DefaultModel()

def register_default_model(model_manager):
    """
    Register the default model with proper configuration.
    """
    # Create a simple default model
    default_model = create_default_model()
    
    # Two options for registering the default model:
    # Option 1: Save the actual model instance (serialized with pickle)
    model_manager.save_model(
        model=default_model,
        model_name="default",
        config={
            "type": "custom",
            "description": "Basic placeholder model for when no other models are available",
            "module": "__main__",  # This module
            "class": "DefaultModel"  # The class name
        }
    )
    
    # Option 2: Register the configuration only (more reliable across sessions)
    model_manager.register_model("default-config", {
        "type": "custom",
        "module": "__main__",
        "class": "DefaultModel",
        "description": "Basic placeholder model registered by configuration"
    })
    
    logger.info("Registered default model")
    return default_model

def parse_args():
    parser = argparse.ArgumentParser(description="AI Model Runner with Enhanced Memory")
    parser.add_argument("--model", type=str, help="Model name to load")
    parser.add_argument("--db-type", type=str, choices=["vector", "sql"], default="vector", 
                        help="Database type to use for memory storage")
    parser.add_argument("--session-id", type=str, help="Resume an existing chat session")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-common-models", action="store_true", help="List common models that can be auto-registered")
    return parser.parse_args()

def handle_list_commands(args, model_manager):
    """Handle list-related commands."""
    if args.list_common_models:
        print("Common models that can be auto-registered:")
        for model_name, config in COMMON_MODELS.items():
            model_type = config["type"]
            model_id = config.get("model_id", model_name)
            print(f"- {model_name} ({model_type}, {model_id})")
        return True
    
    if args.list_models:
        available_models = model_manager.list_available_models()
        if not available_models:
            print("No models available. You can register models using the instructions in the README.")
            print("Using the built-in default model for now.")
        else:
            print("Available models:")
            for model in available_models:
                print(f"- {model}")
        
        print("\nYou can also use any of these common models (they will be auto-registered):")
        for model_name in COMMON_MODELS:
            print(f"- {model_name}")
        return True
    return False

def ensure_default_model(model_manager):
    """Ensure default model is registered."""
    try:
        model_manager.load_model("default")
    except ValueError:
        register_default_model(model_manager)

def handle_model_load(model_name, model_manager):
    """Handle model loading with fallback logic."""
    try:
        return model_manager.load_model(model_name)
    except ValueError as e:
        if model_name == "default":
            logger.info("Creating a fresh default model instance")
            return create_default_model()
        
        if model_name in COMMON_MODELS:
            return handle_common_model(model_name, model_manager)
        
        logger.error(f"Error loading model '{model_name}': {e}")
        print(f"Model '{model_name}' not found. Available models:")
        for model in model_manager.list_available_models():
            print(f"- {model}")
        print("\nCommon models that can be auto-registered:")
        for common_model in COMMON_MODELS:
            print(f"- {common_model}")
        sys.exit(1)

def handle_common_model(model_name, model_manager):
    """Handle loading of common models."""
    print(f"Model '{model_name}' is not registered yet but is available for auto-registration.")
    
    if register_common_model(model_manager, model_name):
        print(f"Successfully registered model: {model_name}")
        try:
            ai_model = model_manager.load_model(model_name)
            print(f"Successfully loaded model: {model_name}")
            return ai_model
        except Exception as e2:
            logger.error(f"Error loading model after registration: {e2}")
            print(f"Failed to load model after registration: {e2}")
    
    print("Using default model instead.")
    return create_default_model()

def run_chat_session(ai_model, memory_manager, session_id):
    """Run the interactive chat session."""
    session = ChatSession(
        model=ai_model,
        memory_manager=memory_manager,
        session_id=session_id
    )
    
    print(f"AI Model Runner initialized with model: {ai_model.__class__.__name__}")
    print("Type 'exit' to end the session, 'help' for commands")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "help":
            print_help()
            continue
            
        response = session.process_input(user_input)
        print(f"AI: {response}")
    
    session.save()
    print("Session saved. Goodbye!")

def main():
    args = parse_args()
    
    # Create necessary directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Initialize managers
    db_manager = DatabaseManager(db_type=args.db_type)
    model_manager = ModelManager()
    memory_manager = MemoryManager(db_manager)
    
    # Handle list commands
    if handle_list_commands(args, model_manager):
        return
    
    # Load specified model or use default
    model_name = args.model or "default"
    
    # Ensure default model is registered
    ensure_default_model(model_manager)
    
    # Load the requested model
    ai_model = handle_model_load(model_name, model_manager)
    
    # Run the chat session
    run_chat_session(ai_model, memory_manager, args.session_id)

def print_help():
    print("\nAvailable commands:")
    print("  help        - Show this help message")
    print("  exit        - End the session and save state")
    print("  !save       - Save the current session")
    print("  !model NAME - Switch to a different model")
    print("  !memory     - Show memory statistics")
    print("  !clear      - Clear conversation context")
    print("\nCommon models you can use:")
    for model_name in COMMON_MODELS:
        print(f"  {model_name}")

if __name__ == "__main__":
    main()
