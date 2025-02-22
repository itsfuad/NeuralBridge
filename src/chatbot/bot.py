from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
import random
from .memory import Memory

class ChatBot:
    def __init__(self):
        self.model_name = "distilgpt2"  # Using a smaller model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.memory_file = "chatbot_memory.json"
        self.learned_responses = self.load_memory()
        self.last_interaction = None
        self.memory = Memory()
        self.conversation_history = []
        self.max_history = 5
        self.system_prompt = "You are a friendly and helpful AI assistant. Keep your responses concise and relevant."
        
    def respond(self, user_input):
        self.last_interaction = user_input.lower()
        
        # Build conversation prompt with system message
        recent_history = self.conversation_history[-self.max_history:] if self.conversation_history else []
        conversation = self.system_prompt + "\n\n" + "\n".join([
            f"Human: {exchange[0]}\nAssistant: {exchange[1]}"
            for exchange in recent_history
        ])
        
        prompt = f"{conversation}\nHuman: {user_input}\nAssistant:"
        
        # Calculate max_length based on input length
        input_length = len(self.tokenizer.encode(prompt))
        max_new_tokens = 50  # Limit new tokens generated
        total_max_length = input_length + max_new_tokens
        
        # Generate response with improved parameters
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,  # Slightly reduced for more focused responses
            top_p=0.9,
            top_k=40,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            length_penalty=1.0,
            early_stopping=True
        )
        
        # Clean and format response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.split('Human:')[0].strip()
        
        # Store in conversation history if response is valid
        if response:
            self.conversation_history.append((user_input, response))
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
        
        return response or "I'm not sure how to respond to that."

    def train(self, user_input, correct_response):
        # Update conversation history with correct response
        if self.conversation_history and self.conversation_history[-1][0] == user_input:
            self.conversation_history[-1] = (user_input, correct_response)
        else:
            self.conversation_history.append((user_input, correct_response))
            
        self.learned_responses[user_input] = correct_response
        self.memory.store(user_input, correct_response, learn=True)
        
    def get_last_interaction(self):
        return self.last_interaction
        
    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.learned_responses, f)
            
    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {}