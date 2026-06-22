from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# 1. 初始化Groq客户端
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. 加载文档
print("正在加载文档...")
documents = []
docs_dir = "docs"
for filename in os.listdir(docs_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(docs_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="gbk") as f:
                content = f.read()
        documents.append(Document(page_content=content, metadata={"source": filename}))

# 3. 文档分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 4. 创建/加载向量数据库
print("正在初始化向量数据库...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 如果chroma_db文件夹已存在就加载，不存在就创建
if os.path.exists("./chroma_db"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    print("已加载现有向量数据库")
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("已创建新的向量数据库")

# 5. 检索相关文档
def retrieve_relevant_docs(query, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    return docs

# 6. 生成回答
def answer_question(query):
    docs = retrieve_relevant_docs(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get("source", "未知") for doc in docs]
    
    prompt = f"""你是一个HR政策助手，根据提供的文档回答用户的问题。
- 只使用上下文中的信息，不要编造
- 如果上下文里没有明确答案，就说"抱歉，我在文档里没有找到相关信息"
- 回答要简洁、专业

上下文：
{context}

用户问题：{query}

回答："""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    return answer, sources

# 7. 交互式对话
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 HR政策助手已启动")
    print("📚 已加载HR政策文档，可以问我关于休假、绩效、远程办公的问题")
    print("💡 输入 'quit' 或 'exit' 退出")
    print("="*50 + "\n")
    
    while True:
        # 获取用户输入
        user_input = input("👤 你：")
        
        # 退出条件
        if user_input.lower() in ["quit", "exit", "退出", "再见"]:
            print("\n🤖 再见！有问题随时来找我~")
            break
        
        if not user_input.strip():
            continue
        
        # 生成回答
        print("🤖 助手：", end="", flush=True)
        answer, sources = answer_question(user_input)
        print(answer)
        print(f"   📄 参考来源：{', '.join(sources)}")
        print()