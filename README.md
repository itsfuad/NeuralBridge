# AI Model Runner System

A flexible system for running AI models with enhanced memory and database integration.

## Features

- **Model Management**: Load, save, and manage AI models from various sources
  - Support for OpenAI API models
  - Support for Hugging Face models
  - Support for custom models
  - Model caching to avoid redundant downloads

- **Extended Context Memory**: Enhanced chat context with persistent memory
  - Session-based conversations
  - Vector search for relevant past messages
  - Metadata attachment to conversations

- **Database Integration**:
  - Vector database support (using FAISS) for semantic search
  - SQL database fallback for simpler setups
  - Automatic switching between databases based on available dependencies

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-model-runner.git
cd ai-model-runner
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

For better performance with FAISS vector database, you might want to install additional packages:
```bash
# For CUDA support (if you have an NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage

Run the system with default settings:

```bash
python main.py
```

This will start a chat session with a default model (if available) or prompt you to configure a model.

### Command-line Arguments

```bash
# Use a specific model
python main.py --model llama2-7b

# Use SQL database instead of vector database
python main.py --db-type sql

# Resume an existing session
python main.py --session-id YOUR_SESSION_ID

# List available models
python main.py --list-models
```

### Interactive Commands

During a chat session, you can use special commands:

- `help` - Show help message
- `!save` - Save the current session
- `!model NAME` - Switch to a different model
- `!memory` - Show memory statistics
- `!clear` - Clear conversation context

### Programmatic Usage

You can also use the system programmatically in your Python code:

```python
from src import ModelManager, MemoryManager, DatabaseManager, ChatSession

# Initialize components
db_manager = DatabaseManager(db_type='vector')
model_manager = ModelManager()
memory_manager = MemoryManager(db_manager)

# Load a model
model = model_manager.load_model('gpt-3.5-turbo')

# Create a chat session
session = ChatSession(
    model=model,
    memory_manager=memory_manager,
    system_message="You are a helpful AI assistant."
)

# Process user input
response = session.process_input("Hello, who are you?")
print(f"AI: {response}")

# Save the session for later
session.save()
session_id = session.get_session_id()
print(f"Session ID: {session_id}")
```

## Adding Your Own Models

### OpenAI Models

```python
model_manager.register_model('gpt-4', {
    'type': 'openai',
    'model_id': 'gpt-4',
    'config': {
        'temperature': 0.7,
        'max_tokens': 1000
    }
})
```

### Hugging Face Models

```python
model_manager.register_model('llama2-7b', {
    'type': 'huggingface',
    'model_id': 'meta-llama/Llama-2-7b-chat-hf',
    'config': {
        'use_auth_token': True  # If needed for access
    }
})
```

### Custom Models

```python
# 1. Create your custom model class
class MyCustomModel:
    def generate(self, messages):
        # Process messages and generate a response
        return "This is a response from my custom model"

# 2. Register the model configuration
model_manager.register_model('my-custom-model', {
    'type': 'custom',
    'module': 'my_module',
    'class': 'MyCustomModel',
    'params': {
        'param1': 'value1'  # Constructor parameters
    }
})

# Or directly save an instantiated model
model = MyCustomModel()
model_manager.save_model(model, 'my-custom-model')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 