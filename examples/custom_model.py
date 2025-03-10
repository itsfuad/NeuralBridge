"""
Custom Model Example

This example demonstrates how to create and use a custom AI model
with the AI Model Runner System.
"""

import os
import sys
import random
import logging

# Add parent directory to path so we can import the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ModelManager, MemoryManager, DatabaseManager, ChatSession

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRandomModel:
    """
    A very simple custom model that provides random responses.
    This is just for demonstration purposes.
    """
    def __init__(self, responses=None):
        """Initialize with optional predefined responses."""
        self.responses = responses or [
            "That's interesting. Tell me more.",
            "I understand. What else is on your mind?",
            "Thank you for sharing that with me.",
            "I'm not sure I follow. Can you elaborate?",
            "That's a great point!",
            "I hadn't thought of it that way before.",
            "Let me think about that for a moment...",
            "I appreciate your perspective on this.",
            "That's a complex issue with many facets.",
            "I'm here to help. What else would you like to know?"
        ]
    
    def generate(self, messages):
        """
        Generate a response based on input messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            A randomly selected response
        """
        # Get the last user message
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        
        if user_messages:
            last_message = user_messages[-1]['content']
            
            # Echo part of the user's message for a more coherent feel
            words = last_message.split()
            if len(words) > 3 and random.random() < 0.3:
                prefix = " ".join(words[:3])
                return f"I noticed you mentioned '{prefix}...'. " + random.choice(self.responses)
        
        # Default to a random response
        return random.choice(self.responses)

def main():
    """Run the example."""
    print("Custom Model Example\n")
    
    # Create the model directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Initialize components
    db_manager = DatabaseManager(db_type='sql', data_dir='data')  # Using SQL for simplicity
    model_manager = ModelManager(models_dir='models')
    memory_manager = MemoryManager(db_manager)
    
    # Create our custom model
    custom_model = SimpleRandomModel()
    
    # Option 1: Save the model directly
    model_path = model_manager.save_model(
        model=custom_model,
        model_name='simple-random',
        config={'type': 'custom', 'description': 'A simple random response model'}
    )
    print(f"Saved custom model to: {model_path}")
    
    # Option 2: Register a model configuration (for models that can't be pickled easily)
    model_manager.register_model('configurable-random', {
        'type': 'custom',
        'module': '__main__',
        'class': 'SimpleRandomModel',
        'params': {
            'responses': [
                "I'm a configurable model with custom responses!",
                "You can specify my responses in the configuration.",
                "This makes me more flexible than a hardcoded model."
            ]
        }
    })
    print("Registered configurable model")
    
    # Now, let's create a chat session with our custom model
    model = model_manager.load_model('simple-random')
    
    session = ChatSession(
        model=model,
        memory_manager=memory_manager,
        system_message="You are a helpful assistant that gives random but relevant responses."
    )
    
    print("\nChat Session Started. Type 'exit' to end.")
    print("--------------------------------------------")
    
    # Interactive chat loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
            
        response = session.process_input(user_input)
        print(f"AI: {response}")
    
    # Save the session before exiting
    session.save()
    session_id = session.get_session_id()
    print(f"\nSession saved with ID: {session_id}")
    print("You can resume this session later with:")
    print(f"python main.py --session-id {session_id} --model simple-random")

if __name__ == "__main__":
    main() 