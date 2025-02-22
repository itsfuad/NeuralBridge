DEFAULT_RESPONSES = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "howdy", "hola"],
        "responses": ["Hello! How can I help you today?", "Hi there!", "Hey! Nice to meet you!"]
    },
    "how_are_you": {
        "patterns": ["how are you", "how are you doing", "how's it going"],
        "responses": ["I'm doing great, thanks for asking!", "I'm fine, how about you?", "All good! What's on your mind?"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "take care"],
        "responses": ["Goodbye!", "See you later!", "Take care!"]
    }
}

DEFAULT_FALLBACK = [
    "I'm not sure I understand. Could you rephrase that?",
    "That's interesting. Tell me more about it.",
    "I'm still learning. Could you elaborate on that?"
]
