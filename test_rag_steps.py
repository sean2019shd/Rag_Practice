print("=== RAG分步测试 ===\n")

# 测试1：文档加载
print("测试1：加载文档...")
try:
    from langchain_community.document_loaders import DirectoryLoader
    loader = DirectoryLoader("docs", glob="**/*.txt")
    documents = loader.load()
    print(f"✅ 成功加载 {len(documents)} 个文档")
    for i, doc in enumerate(documents):
        print(f"   文档{i+1}: {doc.metadata['source']}, 长度: {len(doc.page_content)}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 测试2：文档分块
print("\n测试2：文档分块...")
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 分割成 {len(chunks)} 个文本块")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 测试3：embedding模型
print("\n测试3：加载embedding模型（第一次会下载，可能需要1-2分钟）...")
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("   正在下载/加载模型，请稍候...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ embedding模型加载成功")
    
    # 测试生成一个embedding
    test_embedding = embeddings.embed_query("测试文本")
    print(f"   测试embedding维度: {len(test_embedding)}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 测试4：创建向量数据库
print("\n测试4：创建向量数据库...")
try:
    from langchain_community.vectorstores import Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_test"
    )
    print("✅ 向量数据库创建成功")
    
    # 测试检索
    results = vectorstore.similarity_search("年假有多少天？", k=2)
    print(f"   检索到 {len(results)} 个相关结果")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

print("\n🎉 所有RAG测试通过！")
