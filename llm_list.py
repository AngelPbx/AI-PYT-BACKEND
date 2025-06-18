import time
from openai import OpenAI

client = OpenAI()

start = time.time()
client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "What's the weather today?"}]
)
latency = time.time() - start
print(f"Latency for gpt-4o: {latency:.2f} sec")
