import openai, os

os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"

client = openai.OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
)

resp = client.chat.completions.create(
    model="tinyllama-1.1b-chat-v1.0",
    messages=[{"role": "user", "content": "Hi! Are you connected to LM Studio?"}]
)
print(resp.choices[0].message.content)
