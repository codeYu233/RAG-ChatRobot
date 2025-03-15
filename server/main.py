import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置向量索引保存路径
VECTOR_STORE_PATH = "vector_store.pkl"

# 1. 加载 JSON 知识库
def load_knowledge_base_json(file_path):
    """加载 JSON 格式的知识库"""
    with open(file_path, 'r', encoding='utf-8') as f:
        knowledge_data = json.load(f)
    return [item["content"] for item in knowledge_data]

# 2. 构建向量索引
def build_vector_store(knowledge_base, embedding_model_name='multi-qa-mpnet-base-dot-v1'):
    """将知识库数据转化为向量索引"""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    documents = [Document(page_content=kb) for kb in knowledge_base]
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# 3. 加载或构建向量索引
def load_or_build_vector_store(knowledge_base, vector_store_path=VECTOR_STORE_PATH):
    """从本地加载向量索引文件，如果不存在则重新构建"""
    if os.path.exists(vector_store_path):
        print(f"从本地加载向量索引：{vector_store_path}")
        vector_store = FAISS.load_local(
            vector_store_path,
            HuggingFaceEmbeddings(model_name='multi-qa-mpnet-base-dot-v1'),
            allow_dangerous_deserialization=True  # 添加这个参数
        )
    else:
        print("本地向量索引不存在，正在构建...")
        vector_store = build_vector_store(knowledge_base)
        vector_store.save_local(vector_store_path)  # 保存到本地
        print(f"向量索引已保存到本地：{vector_store_path}")
    return vector_store


# 4. 查询知识库（按归一化分数和动态阈值筛选）
def query_knowledge_base(query, vector_store, top_k=3,threshold=0.5):
    """基于查询从知识库中检索相关内容，并动态筛选相关性较高的结果"""
    docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)  # 检索并获取相似度分数

    # 提取所有分数
    scores = [score for _, score in docs_and_scores]
    min_score, max_score = min(scores), max(scores)

    # 将分数归一化到 [0, 1]，并反转（距离越小分数越高）
    normalized_scores = [1 - (score - min_score) / (max_score - min_score) for score in scores]


    print(f"\n检索到的内容及其相似度分数（标准化到 [0, 1]）：")
    for i, (doc, score) in enumerate(docs_and_scores):
        print(f"内容: {doc.page_content}, 原始分数: {score}, 归一化分数: {normalized_scores[i]}")

    # 筛选分数大于动态阈值的内容
    filtered_docs = [
        doc.page_content
        for i, (doc, _) in enumerate(docs_and_scores)
        if normalized_scores[i] >= threshold and scores[i]<22
    ]

    return filtered_docs


# 5. 加载 GLM-4-9B-Chat 模型
def load_glm_model():
    """加载 glm-4-9b-chat 模型"""
    model_name = "THUDM/glm-4-9b-chat"
    print(f"正在加载模型 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 提高效率
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    return model, tokenizer

# 6. 构建问答流程
def generate_response(query, relevant_knowledge, model, tokenizer, use_knowledge):
    """结合知识库内容生成回答"""
    # 将知识库内容与问题整合
    if use_knowledge and relevant_knowledge:
        background_knowledge = "\n".join(relevant_knowledge)
        prompt = f"以下是相关背景知识：\n{background_knowledge}\n\n用户问题：{query}"
    else:
        prompt = f"用户问题：{query}"

    # 准备输入，使用 GLM 的对话模板
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True)
    inputs = inputs.to(device)

    # 推理生成
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# 7. 主流程
def main():
    # 知识库路径
    knowledge_base_path = "knowledge.json"  # JSON 文件路径

    # 加载知识库
    print("加载知识库...")
    knowledge_base = load_knowledge_base_json(knowledge_base_path)

    # 加载或构建向量索引
    print("加载或构建向量索引...")
    vector_store = load_or_build_vector_store(knowledge_base)

    # 加载 GLM 模型
    print("加载 GLM-4-9B-Chat 模型...")
    model, tokenizer = load_glm_model()

    # 用户输入
    while True:
        query = input("\n请输入：")
        if query.lower() == "exit":
            print("退出程序。")
            break
        
        # 询问用户是否使用知识库
        use_knowledge_input = input("是否使用知识库回答（yes/no）？").strip().lower()
        use_knowledge = use_knowledge_input == "yes"
        
        # 从知识库中检索相关知识（如果选择使用知识库）
        relevant_knowledge = []
        if use_knowledge:
            relevant_knowledge = query_knowledge_base(query, vector_store)
            print("\n检索到的相关知识：", relevant_knowledge if relevant_knowledge else "无相关知识符合相似度要求")
        
        # 生成回答
        response = generate_response(query, relevant_knowledge, model, tokenizer, use_knowledge)
        print("\n生成的回答：", response)

if __name__ == "__main__":
    main()
