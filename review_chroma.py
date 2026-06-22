print("开始检查Chroma数据库...\n")

# 测试1：检查文件夹是否存在
import os
print("1. 检查chroma_db文件夹...")
if os.path.exists("./chroma_db"):
    print(f"   ✅ 文件夹存在")
    print(f"   内容: {os.listdir('./chroma_db')}")
else:
    print("   ❌ 文件夹不存在")
    exit()

# 测试2：导入库
print("\n2. 导入库...")
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("   ✅ 导入成功")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    exit()

# 测试3：加载embedding
print("\n3. 加载embedding模型（可能需要几秒钟）...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("   ✅ 模型加载成功")
except Exception as e:
    print(f"   ❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 测试4：加载向量数据库
print("\n4. 加载向量数据库...")
try:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    print("   ✅ 数据库加载成功")
except Exception as e:
    print(f"   ❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 测试5：获取所有文档
print("\n5. 获取文档内容...")
try:
    all_docs = vectorstore.get()
    print(f"   ✅ 共有 {len(all_docs['documents'])} 个文本块\n")
    
    for i, (doc, meta) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
        print(f"--- 文本块 {i+1} ---")
        print(f"来源：{meta.get('source', '未知')}")
        print(f"内容：{doc[:100]}...")
        print()
        
except Exception as e:
    print(f"   ❌ 获取失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

print("✅ 检查完成！")