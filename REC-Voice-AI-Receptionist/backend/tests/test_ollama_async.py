import asyncio

from openai import AsyncOpenAI


async def main():

    client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        timeout=120.0,
        max_retries=0,
    )

    print("Connecting to Ollama...")

    response = await client.chat.completions.create(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": "Say OK",
            }
        ],
        max_tokens=20,
    )

    print()
    print("OLLAMA TEST RESULT:")
    print(response.choices[0].message.content)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())