# main.py

from chatbot.bot import ChatBot

def main():
    chatbot = ChatBot()
    print("Welcome to the ChatBot! You can start chatting now.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("ChatBot: Goodbye!")
            break
        response = chatbot.respond(user_input)
        print(f"ChatBot: {response}")

if __name__ == "__main__":
    main()
