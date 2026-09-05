import ollama
def main():
    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise AI receptionist "
                    "for Rajalakshmi Engineering College."
                ),
            },
            {
                "role": "user",
                "content": "Say hello in one sentence.",
            },
        ],
    )
    print("\nAI RESPONSE:")
    print(response["message"]["content"])
if __name__ == "__main__":
    main()