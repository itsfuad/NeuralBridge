class Memory:
    def __init__(self):
        self.conversations = []
        self.knowledge_base = {}

    def remember(self, user_input, bot_response):
        self.conversations.append((user_input, bot_response))
        self.update_knowledge_base(user_input, bot_response)

    def update_knowledge_base(self, user_input, bot_response):
        # This method can be expanded to include more sophisticated memory management
        self.knowledge_base[user_input] = bot_response

    def recall(self):
        return self.conversations

    def get_response_from_memory(self, user_input):
        return self.knowledge_base.get(user_input, None)