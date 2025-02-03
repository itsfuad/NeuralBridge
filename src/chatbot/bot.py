from langdetect import detect
from transformers import AutoModelForCausalLM, AutoTokenizer
from chatbot.memory import Memory
from chatbot.trainer import Trainer

class ChatBot:
    def __init__(self):
        self.memory = Memory()
        self.trainer = Trainer(self.memory)
        model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def understand(self, user_input):
        # Process the input to understand the intent and context
        # This can include language detection for Bangla and English
        language = detect(user_input)
        return language

    def respond(self, user_input):
        # Generate a response based on the understood input
        # Use memory to recall past conversations if necessary
        inputs = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return response

    def remember(self, information):
        # Store relevant information in memory
        self.memory.store(information)

    def learn(self, new_data):
        # Train the chatbot with new data
        self.trainer.train(new_data)
