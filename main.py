import sqlite3
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatBot:
    def __init__(self):
        # Initialize language model
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize database
        self.conn = sqlite3.connect('chatbot_memory.db')
        self.create_tables()
        
        # Context management
        self.context_window = []
        self.max_context_length = 4  # Remember last 4 exchanges
        
        # Learning parameters
        self.learning_threshold = 2  # Number of mentions to remember
        
    def create_tables(self):
        cursor = self.conn.cursor()
        # Create memories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT,
                value TEXT,
                timestamp DATETIME,
                context TEXT
            )
        ''')
        
        # Create conversation log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                bot_response TEXT,
                timestamp DATETIME
            )
        ''')
        self.conn.commit()

    def store_memory(self, key, value, context):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO memories (key, value, timestamp, context)
            VALUES (?, ?, ?, ?)
        ''', (key.lower(), value, datetime.now(), context))
        self.conn.commit()

    def retrieve_memory(self, key):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT value, context FROM memories 
            WHERE key = ? 
            ORDER BY timestamp DESC
        ''', (key.lower(),))
        return cursor.fetchall()

    def detect_intent(self, user_input):
        # Simple intent detection using pattern matching
        lower_input = user_input.lower()
        
        if any(word in lower_input for word in ['remember', 'store', 'save']):
            return 'store_memory'
        elif any(word in lower_input for word in ['what', 'tell', 'know']):
            return 'retrieve_memory'
        else:
            return 'general_chat'

    def generate_response(self, prompt, max_length=100):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def update_context(self, user_input, response):
        self.context_window.append((user_input, response))
        if len(self.context_window) > self.max_context_length:
            self.context_window.pop(0)

    def process_input(self, user_input):
        # Add previous context to input
        context = "\n".join([f"User: {u}\nBot: {r}" for u, r in self.context_window])
        full_prompt = f"{context}\nUser: {user_input}\nBot:"
        
        # Detect user intent
        intent = self.detect_intent(user_input)
        
        if intent == 'store_memory':
            # Basic pattern matching for memory storage
            if 'is' in user_input:
                key, value = user_input.split('is', 1)
                self.store_memory(key.strip(), value.strip(), context)
                return "I'll remember that!"
            
        elif intent == 'retrieve_memory':
            # Search for keywords to retrieve memory
            key = user_input.replace('?', '').split()[-1]
            memories = self.retrieve_memory(key)
            if memories:
                return f"I remember: {memories[0][0]} (from {memories[0][1]})"
            else:
                return "I don't have information about that yet."
        
        # Generate response using LLM
        response = self.generate_response(full_prompt)
        # Remove the prompt from response
        response = response[len(full_prompt):].split('\n')[0].strip()
        
        # Update conversation history
        self.update_context(user_input, response)
        self.log_conversation(user_input, response)
        
        return response

    def log_conversation(self, user_input, response):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_input, bot_response, timestamp)
            VALUES (?, ?, ?)
        ''', (user_input, response, datetime.now()))
        self.conn.commit()

# Main loop
if __name__ == "__main__":
    bot = ChatBot()
    print("ChatBot: Hello! How can I help you today? (Type 'exit' to end)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = bot.process_input(user_input)
        print(f"ChatBot: {response}")