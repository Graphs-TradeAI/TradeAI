import os
import sys
from dotenv import load_dotenv
from groq import Groq

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def test_groq():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env")
        return

    client = Groq(api_key=api_key)
    model = "llama-3.3-70b-versatile"
    
    print(f"Testing Groq with model: {model}")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say hello in one word",
                }
            ],
            model=model,
        )
        print("Groq Response:", chat_completion.choices[0].message.content)
        print("Groq Connection: SUCCESS")
    except Exception as e:
        print(f"Groq Connection: FAILED - {e}")

if __name__ == "__main__":
    test_groq()
