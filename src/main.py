# main.py

from chatbot.bot import ChatBot
from chatbot.initialize import download_nltk_resources

def main():
    # Download required NLTK resources
    download_nltk_resources()
    
    chatbot = ChatBot()
    print("ChatBot: Hello! Let's chat. Use ':train' to correct my responses.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            chatbot.save_memory()
            print("ChatBot: Goodbye!")
            break
            
        if user_input.lower() == ":train":
            previous_input = chatbot.get_last_interaction()
            if previous_input:
                print(f"Original input was: {previous_input}")
                correct_response = input("What should I have said?: ")
                chatbot.train(previous_input, correct_response)
                print("ChatBot: Thanks! I'll try to do better next time.")
            continue
        
        response = chatbot.respond(user_input)
        print(f"ChatBot: {response}")

if __name__ == "__main__":
    main()