from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 用Groq的API，兼容OpenAI格式
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
    #model="llama3-8b-8192",  # Groq的免费模型
    model="llama-3.1-8b-instant",  # 改成这个新模型名
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)

print(response.choices[0].message.content)

