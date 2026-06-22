print("开始运行...")

try:
    print("1. 导入库...")
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    print("   导入成功")
except Exception as e:
    print(f"   导入失败: {e}")
    exit()

try:
    print("2. 加载环境变量...")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    print(f"   API Key长度: {len(api_key) if api_key else 0}")
except Exception as e:
    print(f"   加载失败: {e}")
    exit()

try:
    print("3. 初始化Groq客户端...")
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
    print("   初始化成功")
except Exception as e:
    print(f"   初始化失败: {e}")
    exit()

try:
    print("4. 测试API调用...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hi"}]
    )
    print(f"   回答: {response.choices[0].message.content}")
except Exception as e:
    print(f"   API调用失败: {e}")
    exit()

print("\n✅ 基础测试全部通过！")