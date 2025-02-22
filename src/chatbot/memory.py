from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import json
import os

class Memory:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.conversations = []
        self.embeddings = []
        self.responses = []
        self.response_patterns = defaultdict(list)
        self.learning_threshold = 0.85
        self.patterns_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "learned_patterns.json")
        self.load_learned_patterns()
        self.weights = {}

    def store(self, user_input, bot_response, learn=False):
        self.conversations.append((user_input, bot_response))
        
        if learn:
            # Create and store embeddings
            embedding = self.get_embeddings(user_input)
            self.embeddings.append(embedding)
            self.responses.append(bot_response)
            
            # Learn patterns
            self.learn_patterns(user_input, bot_response)

    def get_embeddings(self, text):
        return self.embedding_model.encode([text])[0]

    def learn_patterns(self, user_input, response):
        # Extract key patterns and associate them with responses
        words = user_input.lower().split()
        embedding = self.get_embeddings(user_input)
        
        for word in words:
            self.response_patterns[word].append({
                'response': response,
                'embedding': embedding,
                'frequency': 1
            })

    def find_similar_response(self, user_input):
        if not self.embeddings:
            return None
            
        query_embedding = self.get_embeddings(user_input)
        similarities = [
            np.dot(query_embedding, stored_embedding) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
            for stored_embedding in self.embeddings
        ]
        
        max_similarity_idx = np.argmax(similarities)
        if similarities[max_similarity_idx] > self.learning_threshold:
            return self.responses[max_similarity_idx]
        return None

    def find_similar_patterns(self, embedding, similarity_threshold=0.8):
        """Find patterns similar to the given embedding"""
        similar_patterns = []
        
        for patterns in self.response_patterns.values():
            for pattern in patterns:
                if 'embedding' in pattern:
                    stored_embedding = pattern['embedding']
                    similarity = np.dot(embedding, stored_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                    )
                    if similarity > similarity_threshold:
                        similar_patterns.append(pattern)
        
        return similar_patterns

    def update_response_patterns(self, embedding, response):
        """Update response patterns based on new learning"""
        similar_patterns = self.find_similar_patterns(embedding)
        if similar_patterns:
            for pattern in similar_patterns:
                pattern['frequency'] += 1
        else:
            # If no similar patterns found, create a new one
            new_pattern = {
                'response': response,
                'embedding': embedding,
                'frequency': 1
            }
            key = response.split()[0] if response else 'default'
            self.response_patterns[key].append(new_pattern)

    def calculate_confidence(self, embedding):
        if not self.embeddings:
            return 0.0
        
        similarities = [
            np.dot(embedding, stored_embedding) / 
            (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
            for stored_embedding in self.embeddings
        ]
        return max(similarities)

    def reinforce_learning(self, interaction, was_good):
        """Reinforce learning based on feedback"""
        embedding = self.get_embeddings(interaction)
        key = str(embedding.tobytes())
        
        if key in self.weights:
            adjustment = 0.1 if was_good else -0.05
            self.weights[key] += adjustment
            self.weights[key] = max(0.1, min(2.0, self.weights[key]))

    def save_learned_patterns(self):
        """Save learned patterns to file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.patterns_file), exist_ok=True)
        
        patterns_to_save = {
            'weights': {k: float(v) for k, v in self.weights.items()},
            'response_patterns': {}
        }
        
        # Convert response patterns to serializable format
        for key, patterns in self.response_patterns.items():
            patterns_to_save['response_patterns'][key] = []
            for pattern in patterns:
                serializable_pattern = {
                    'response': pattern['response'],
                    'embedding': pattern['embedding'].tolist() if isinstance(pattern['embedding'], np.ndarray) else pattern['embedding'],
                    'frequency': pattern['frequency']
                }
                patterns_to_save['response_patterns'][key].append(serializable_pattern)
        
        try:
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_to_save, f)
        except Exception as e:
            print(f"Warning: Could not save patterns file: {e}")

    def load_learned_patterns(self):
        """Load previously learned patterns"""
        # Initialize empty defaults
        self.weights = {}
        self.response_patterns = defaultdict(list)
        
        if not os.path.exists(self.patterns_file):
            return

        try:
            with open(self.patterns_file, 'r') as f:
                patterns = json.load(f)
                
                # Load weights if they exist
                if 'weights' in patterns:
                    self.weights = {k: float(v) for k, v in patterns['weights'].items()}
                
                # Load response patterns if they exist
                if 'response_patterns' in patterns:
                    for key, pattern_list in patterns['response_patterns'].items():
                        self.response_patterns[key] = []
                        for pattern in pattern_list:
                            if isinstance(pattern, dict) and 'embedding' in pattern:
                                pattern['embedding'] = np.array(pattern['embedding']) if pattern['embedding'] else None
                                self.response_patterns[key].append(pattern)
        except (Exception) as e:
            print(f"Warning: Could not load patterns file: {e}")
            # File is corrupted, create a backup
            if os.path.exists(self.patterns_file):
                backup_file = f"{self.patterns_file}.bak"
                try:
                    os.rename(self.patterns_file, backup_file)
                    print(f"Corrupted patterns file backed up to: {backup_file}")
                except Exception as e:
                    print(f"Warning: Could not create backup file: {e}")