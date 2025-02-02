# Chatbot Application

This project is a chatbot application that can understand and converse in both Bangla and English. It is designed to learn from conversations and improve its responses over time.

## Features

- Natural language understanding in Bangla and English
- Memory management to remember past conversations
- Self-training capabilities to improve over time

## Project Structure

```
chatbot-app
├── src
│   ├── main.py          # Entry point of the application
│   ├── chatbot
│   │   ├── __init__.py  # Package initialization
│   │   ├── bot.py       # ChatBot class for handling conversations
│   │   ├── memory.py    # Memory class for storing past interactions
│   │   └── trainer.py    # Trainer class for self-training
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── .gitignore            # Files to ignore in version control
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd chatbot-app
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the chatbot, execute the following command:
```
python src/main.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License

This project is licensed under the MIT License.