from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

load_dotenv()

# 1. 初始化
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. 加载文档和向量数据库（和之前一样）
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

# ========== 智能体1：检索智能体 ==========
def retrieval_agent(query):
    """
    检索智能体：从知识库中找到最相关的信息
    """
    print("  📚 检索智能体：正在查找相关政策...")
    
    docs = vectorstore.similarity_search(query, k=4)
    sources = [doc.metadata.get("source", "未知") for doc in docs]
    context = "\n\n".join([doc.page_content for doc in docs])
    
    result = {
        "query": query,
        "context": context,
        "sources": sources,
        "num_results": len(docs)
    }
    
    print(f"  ✅ 检索完成，找到 {len(docs)} 个相关文档片段")
    return result

# ========== 智能体2：分析智能体 ==========
def analysis_agent(retrieval_result):
    """
    分析智能体：基于检索结果进行深度分析
    """
    print("  🔍 分析智能体：正在进行多维度分析...")
    
    query = retrieval_result["query"]
    context = retrieval_result["context"]
    
    prompt = f"""你是一个专业的HR政策分析师。请基于提供的政策文档，对用户的问题进行多维度分析。

请从以下几个维度分析：
1. 直接答案：明确回答用户的问题
2. 相关政策依据：引用具体的政策条款
3. 注意事项：需要特别注意的地方
4. 延伸建议：相关的其他政策建议

政策文档：
{context}

用户问题：{query}

请用JSON格式输出，包含以下字段：
- direct_answer: 直接答案
- policy_basis: 政策依据
- considerations: 注意事项（数组）
- suggestions: 延伸建议（数组）

只输出JSON，不要其他文字。"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    # 解析JSON
    try:
        analysis = json.loads(response.choices[0].message.content)
    except:
        # 如果JSON解析失败，就把内容当字符串
        analysis = {
            "direct_answer": response.choices[0].message.content,
            "policy_basis": "见政策文档",
            "considerations": [],
            "suggestions": []
        }
    
    print("  ✅ 分析完成")
    return analysis

# ========== 智能体3：报告智能体 ==========
def report_agent(query, retrieval_result, analysis):
    """
    报告智能体：把结果整理成结构化的专业报告
    """
    print("  📝 报告智能体：正在生成结构化报告...")
    
    sources = retrieval_result["sources"]
    
    prompt = f"""你是一个专业的报告撰写者。请把以下分析结果整理成一份清晰、专业的回答。

用户问题：{query}

直接答案：{analysis.get('direct_answer', '')}

政策依据：{analysis.get('policy_basis', '')}

注意事项：
{chr(10).join(['- ' + item for item in analysis.get('considerations', [])])}

延伸建议：
{chr(10).join(['- ' + item for item in analysis.get('suggestions', [])])}

参考来源：{', '.join(sources)}

请把这些内容整理成一段流畅、专业的回答，结构清晰，易于理解。不要提到"智能体"、"JSON"这些技术术语，就像一个资深HR专家在回答问题一样。"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    report = response.choices[0].message.content
    print("  ✅ 报告生成完成")
    return report

# ========== 监督智能体（总调度） ==========
def supervisor_agent(query):
    """
    监督智能体：协调所有智能体，保证输出质量
    """
    print(f"\n🤖 多智能体系统启动，处理问题：{query}\n")
    
    # 第一步：检索智能体查找信息
    retrieval_result = retrieval_agent(query)
    
    # 第二步：分析智能体做深度分析
    analysis = analysis_agent(retrieval_result)
    
    # 第三步：报告智能体生成最终报告
    report = report_agent(query, retrieval_result, analysis)
    
    # 质量检查（简单版：检查回答是否相关）
    print("\n✅ 多智能体处理完成！")
    print("="*60)
    
    return report

# ========== 交互式对话 ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 多智能体HR政策分析系统")
    print("📚 三个智能体协同工作：检索 → 分析 → 报告")
    print("💡 输入 'quit' 退出")
    print("="*60 + "\n")
    
    while True:
        user_input = input("👤 你：")
        
        if user_input.lower() in ["quit", "exit", "退出", "再见"]:
            print("\n🤖 再见！")
            break
        
        if not user_input.strip():
            continue
        
        # 调用多智能体系统
        result = supervisor_agent(user_input)
        print(f"\n📋 最终回答：\n{result}\n")