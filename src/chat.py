"""
Chat Session Module

This module handles the interaction with AI models.
It manages the conversation flow, context retrieval, and model responses.
"""

import logging
import uuid
import datetime
import json
import re
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatSession:
    """Manages chat interactions with an AI model."""
    
    def __init__(self, model: Any, memory_manager, session_id: Optional[str] = None, 
                 system_message: str = None):
        """
        Initialize the chat session.
        
        Args:
            model: AI model to use for the session
            memory_manager: Memory manager instance
            session_id: Existing session ID to resume, or None to create a new session
            system_message: Optional system message to set the behavior of the AI
        """
        self.model = model
        self.memory_manager = memory_manager
        self.system_message = system_message or "You are a helpful assistant."
        
        # Create or load session
        if session_id:
            try:
                # Try to load existing session
                self.session_data = memory_manager.load_session(session_id)
                self.session_id = session_id
                logger.info(f"Resumed existing session: {session_id}")
            except ValueError:
                # If session doesn't exist, create a new one
                self.session_id = self._create_new_session()
        else:
            # Create a new session
            self.session_id = self._create_new_session()
    
    def _create_new_session(self) -> str:
        """Create a new chat session."""
        model_name = None
        if isinstance(self.model, dict) and 'model_id' in self.model:
            model_name = self.model['model_id']
        
        session_id = self.memory_manager.create_session(model_name=model_name)
        
        # Add system message if provided
        if self.system_message:
            self.memory_manager.add_message(
                session_id=session_id,
                role="system",
                content=self.system_message
            )
        
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and get a response from the AI model.
        
        Args:
            user_input: User input text
            
        Returns:
            AI model response
        """
        # Check for special commands
        if user_input.startswith('!'):
            return self._handle_command(user_input)
        
        # Add user message to memory
        self.memory_manager.add_message(
            session_id=self.session_id,
            role="user",
            content=user_input
        )
        
        # Get conversation history
        messages = self._prepare_conversation_context(user_input)
        
        # Generate response from the model
        try:
            response = self._get_model_response(messages)
            
            # Add assistant response to memory
            self.memory_manager.add_message(
                session_id=self.session_id,
                role="assistant",
                content=response
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _prepare_conversation_context(self, user_input: str) -> List[Dict]:
        """
        Prepare the conversation context for the model.
        
        This includes:
        1. Recent conversation history
        2. Relevant messages from past conversations
        
        Args:
            user_input: Current user input
            
        Returns:
            List of messages in the conversation context
        """
        # Start with system message
        context = [
            {"role": "system", "content": self.system_message}
        ]
        
        # Add recent conversation history (last 10 messages)
        recent_messages = self.memory_manager.get_messages(self.session_id, limit=10)
        
        # Get relevant context from past conversations
        relevant_context = self.memory_manager.get_context(self.session_id, user_input, limit=3)
        
        # If we have relevant context, add it to the system message
        if relevant_context:
            context_str = "\n\nRelevant information from past conversations:\n"
            for i, msg in enumerate(relevant_context):
                context_str += f"[{i+1}] {msg['role'].capitalize()}: {msg['content']}\n"
            
            # Add to system message
            context[0]["content"] += context_str
        
        # Add recent conversation messages
        for msg in recent_messages:
            if msg['role'] in ['user', 'assistant', 'system']:
                context.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        return context
    
    def _get_model_response(self, messages: List[Dict]) -> str:
        """
        Get a response from the AI model.
        
        This method handles different types of models:
        - OpenAI API models
        - Hugging Face models
        - Custom models
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Model response text
        """
        model = self.model
        
        # Handle OpenAI models
        if isinstance(model, dict) and model.get('type') == 'openai':
            return self._get_openai_response(model, messages)
        
        # Handle Hugging Face models
        elif isinstance(model, dict) and 'model' in model and 'tokenizer' in model:
            return self._get_huggingface_response(model, messages)
        
        # Handle custom models
        else:
            return self._get_custom_model_response(model, messages)
    
    def _get_openai_response(self, model_data: Dict, messages: List[Dict]) -> str:
        """Get a response from an OpenAI model."""
        try:
            import openai
            
            # Configure OpenAI client
            model_id = model_data.get('model_id', 'gpt-3.5-turbo')
            
            # Make API request
            response = openai.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=model_data.get('config', {}).get('temperature', 0.7),
                max_tokens=model_data.get('config', {}).get('max_tokens', 1000)
            )
            
            # Extract and return the response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return "No response generated."
                
        except ImportError:
            logger.error("OpenAI package not installed.")
            return "Error: OpenAI package not installed. Please install it with 'pip install openai'."
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return f"Error: {str(e)}"
    
    def _get_huggingface_response(self, model_data: Dict, messages: List[Dict]) -> str:
        """Get a response from a Hugging Face model."""
        try:
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            config = model_data.get('config', {})
            
            # Convert messages to a single text prompt
            prompt = self._format_messages_for_huggingface(messages)
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            # Generate
            outputs = model.generate(
                inputs.input_ids,
                max_length=config.get('max_length', 1000),
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 0.9),
                num_beams=config.get('num_beams', 4),
                do_sample=config.get('do_sample', True)
            )
            
            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the model's reply from the full response
            response = self._extract_model_reply(prompt, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error with Hugging Face model: {e}")
            return f"Error with model generation: {str(e)}"
    
    def _format_messages_for_huggingface(self, messages: List[Dict]) -> str:
        """Format chat messages for a Hugging Face model."""
        formatted_prompt = ""
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                formatted_prompt += f"System: {content}\n\n"
            elif role == 'user':
                formatted_prompt += f"User: {content}\n"
            elif role == 'assistant':
                formatted_prompt += f"Assistant: {content}\n"
        
        # Add a final assistant prompt
        formatted_prompt += "Assistant: "
        
        return formatted_prompt
    
    def _extract_model_reply(self, prompt: str, full_response: str) -> str:
        """Extract the model's reply from the full response."""
        # If the response has more content than the prompt,
        # extract just the new content
        if full_response.startswith(prompt):
            return full_response[len(prompt):].strip()
        
        # Try to find where the assistant's response starts
        pattern = r"Assistant: (.*)(?:User:|System:|$)"
        matches = re.findall(pattern, full_response, re.DOTALL)
        
        if matches and len(matches) > 0:
            return matches[-1].strip()  # Return the last assistant response
        
        return full_response.strip()
    
    def _get_custom_model_response(self, model: Any, messages: List[Dict]) -> str:
        """Get a response from a custom model."""
        try:
            # Check if model has a specific chat method
            if hasattr(model, 'chat') and callable(model.chat):
                return model.chat(messages)
            
            # Check if model has a generate method that accepts messages
            elif hasattr(model, 'generate') and callable(model.generate):
                return model.generate(messages)
            
            # Otherwise, fall back to a simple generate method with text prompt
            elif hasattr(model, 'generate_text') and callable(model.generate_text):
                prompt = self._format_messages_for_huggingface(messages)
                return model.generate_text(prompt)
            
            else:
                return "Error: Model does not have a compatible interface for chat."
                
        except Exception as e:
            logger.error(f"Error with custom model: {e}")
            return f"Error with model generation: {str(e)}"
    
    def _handle_command(self, command: str) -> str:
        """
        Handle special commands.
        
        Commands start with ! and include:
        - !save - Save the current session
        - !model NAME - Switch to a different model
        - !memory - Show memory statistics
        - !clear - Clear conversation context
        
        Args:
            command: Command string
            
        Returns:
            Command response
        """
        # Split command and arguments
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        
        # Handle commands
        if cmd == '!save':
            self.save()
            return "Session saved."
            
        elif cmd == '!model':
            # Model switching would need to be implemented in the main app
            return "Model switching can be done by restarting with --model parameter."
            
        elif cmd == '!memory':
            # Show memory statistics
            stats = self._get_memory_stats()
            return f"Memory statistics:\n{stats}"
            
        elif cmd == '!clear':
            # Clear conversation context
            self.memory_manager.clear_context(self.session_id)
            return "Conversation context cleared."
            
        else:
            return f"Unknown command: {cmd}. Type 'help' for a list of commands."
    
    def _get_memory_stats(self) -> str:
        """Get memory statistics for the current session."""
        try:
            # Get session summary
            summary = self.memory_manager.get_session_summary(self.session_id)
            
            # Format statistics
            stats = []
            stats.append(f"Session ID: {summary['id']}")
            stats.append(f"Created: {summary['created_at']}")
            stats.append(f"Updated: {summary.get('updated_at', 'Never')}")
            stats.append(f"Model: {summary.get('model', 'Unknown')}")
            stats.append(f"Message count: {summary['message_count']}")
            
            return "\n".join(stats)
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return f"Error retrieving memory statistics: {str(e)}"
    
    def save(self) -> None:
        """Save the current session state."""
        self.memory_manager.save_session(self.session_id)
        logger.info(f"Saved session: {self.session_id}")
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id 