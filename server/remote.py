import argparse
from main import (
    load_knowledge_base_json,
    load_or_build_vector_store,
    load_glm_model,
    k_hop_search,
    generate_response,
)



def run_server():
    """启动服务器模式"""
    knowledge_base_path = "knowledge.json"  # JSON 文件路径
    knowledge_base = load_knowledge_base_json(knowledge_base_path)
    vector_store = load_or_build_vector_store(knowledge_base)
    model, tokenizer = load_glm_model()
    print("模型已启动，等待输入...")
    while True:
        try:
            # 从命令行读取输入
            user_input = input()
            if user_input.lower() == "exit":
                print("服务器已停止。")
                break

            # 解析输入
            parts = user_input.split("|")
            if len(parts) == 3:
                query = parts[0]
                k = int(parts[1].strip())
                use_knowledge = parts[2].strip().lower() == "true"
            else:
                print("输入格式不正确")
                continue

            # 检索知识并生成回答
            relevant_knowledge = []
            if use_knowledge:
                relevant_knowledge = k_hop_search(query, vector_store, k=k)
                print("\n检索到的相关知识：", relevant_knowledge if relevant_knowledge else "无相关知识符合相似度要求")
            response = generate_response(query, relevant_knowledge, model, tokenizer, use_knowledge)
            print("\n", response)
            print("---END---")
        except Exception as e:
            print(f"发生错误：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="服务器端脚本")
    parser.add_argument("--start", action="store_true", help="启动服务器模式")
    args = parser.parse_args()

    if args.start:
        run_server()
