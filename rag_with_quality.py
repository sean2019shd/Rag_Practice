from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# 1. 初始化
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. 加载和处理文档（和之前一样）
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 3. 带质量评估的问答
def answer_question_with_quality(query, confidence_threshold=0.5):
    """
    带质量评估的RAG问答
    返回：回答、引用来源、置信度、是否需要人工审核
    """
    
    # 检索相关文档（带分数）
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    # 计算置信度（基于最高相似度分数）
    # 注意：Chroma返回的是距离，越小越相似，我们转换成0-1的置信度
    if results:
        max_score = results[0][1]
        # 简单的距离转置信度（距离越小，置信度越高）
        confidence = max(0, min(1, 1.0 - max_score / 2.0))
    else:
        confidence = 0.0
    
    # 提取文档内容和来源
    docs = [r[0] for r in results]
    sources = [r[0].metadata.get("source", "未知") for r in results]
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 判断是否需要人工审核
    needs_human_review = confidence < confidence_threshold
    
    # 构建prompt（要求模型只基于上下文回答）
    prompt = f"""根据以下上下文回答用户的问题。
- 只使用上下文中的信息，不要编造
- 如果上下文里没有明确答案，就说"抱歉，我在文档里没有找到相关信息"
- 回答要简洁准确

上下文：
{context}

用户问题：{query}

回答："""
    
    # 调用LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence, 2),
        "needs_human_review": needs_human_review,
        "context": context
    }

# 4. 测试
if __name__ == "__main__":
    print("=== RAG质量评估测试 ===\n")
    
    test_questions = [
        "入职第3年有多少天年假？",
        "绩效评估有几个等级？",
        "远程办公每月有多少补贴？",
        "公司的CEO是谁？",  # 文档里没有的问题，应该低置信度
    ]
    
    for q in test_questions:
        result = answer_question_with_quality(q)
        
        print(f"问题：{q}")
        print(f"回答：{result['answer']}")
        print(f"置信度：{result['confidence']}")
        print(f"引用来源：{', '.join(result['sources'])}")
        print(f"需要人工审核：{'是 ⚠️' if result['needs_human_review'] else '否 ✅'}")
        print()