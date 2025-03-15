import gradio as gr
import paramiko
import time

# SSH 配置(填入服务器信息)
SSH_HOST = ""
SSH_PORT = ""
SSH_USER = ""
SSH_PASSWORD = ""

# 全局变量
ssh_client = None
server_channel = None

def connect_to_server():
    """连接到服务器"""
    global ssh_client, server_channel
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 使用密码进行连接
    ssh_client.connect(
        SSH_HOST,
        port=SSH_PORT,
        username=SSH_USER,
        password=SSH_PASSWORD
    )

    server_channel = ssh_client.invoke_shell()
    time.sleep(1)
    server_channel.send("cd autodl-tmp\n")  # 确保切换到脚本所在目录（这里以autodl服务器为例）
    server_channel.send("source /etc/network_turbo\n")# autodl服务器访问huggingface代理指令，根据实际情况选择
    time.sleep(1)

def start_main_script():
    """启动 main.py 并实时接收服务器输出"""
    global server_channel

    # 发送启动命令
    server_channel.send("python remote.py --start\n")  

    # 初始化输出缓冲区
    output_lines = []  

    # 循环接收服务器返回的信息
    while True:
        if server_channel.recv_ready():
            # 读取输出并解码
            partial_output = server_channel.recv(1024).decode('utf-8')
            print(partial_output, end="")  # 实时打印到本地控制台

            # 按行拆分新接收到的输出
            for line in partial_output.splitlines():
                output_lines.append(line)
                # 保持缓冲区中仅存储最新的两行
                if len(output_lines) > 1:
                    output_lines.pop(0)

            # 检测模型启动完成的标志性输出
            if "模型已启动，等待输入..." in partial_output:  
                break

        time.sleep(0.5)  # 等待片刻，避免过度占用资源

    # 返回最新的两行内容
    return "\n".join(output_lines).strip()

def stop_server():
    """停止服务器"""
    global server_channel
    try:
        if server_channel:
            server_channel.send("exit\n")  # 停止模型或服务器脚本
            time.sleep(1)  # 等待模型停止
            return "模型已停止！"
        else:
            return "未连接到服务器！"
    except Exception as e:
        return f"停止失败：{e}"

def send_query(query, k=1, use_knowledge=True, state=[]):
    """发送查询到服务器并接收响应"""
    global server_channel
    end_marker = "---END---"
    try:
        if server_channel is None or server_channel.closed:
            return "服务器未连接，请先启动模型！"

        # 格式化查询消息并发送
        message = f"{query}|{k}|{'true' if use_knowledge else 'false'}\n"
        server_channel.send(message)

        # 初始化缓冲区
        output = ""
        buffer = bytearray()  # 用于存储未完整接收的字节

        while True:
            chunk = server_channel.recv(1024)  # 接收二进制数据
            buffer.extend(chunk)  # 添加到缓冲区

            try:
                # 尝试解码缓冲区数据
                output += buffer.decode('utf-8')
                buffer.clear()  # 如果成功解码，清空缓冲区
            except UnicodeDecodeError:
                # 如果解码失败，说明数据不完整，继续接收更多数据
                continue

            # 检查是否包含结束标志
            if end_marker in output:
                break

        # 删除结束标志
        output = output.replace(end_marker, "").strip()

        # 删除第一行
        output_lines = output.splitlines()
        if len(output_lines) > 1:
            output = "\n".join(output_lines[1:])  # 保留从第二行开始的内容
        else:
            output = ""  # 如果只有一行，返回空字符串

        # 更新聊天历史
        state.append((query,output.strip()))  # 添加用户的消息
        return state  # 返回更新后的聊天历史

    except Exception as e:
        return f"查询失败：{e}"

def start_server():
    """启动模型并返回初始输出"""
    try:
        if ssh_client is None or not ssh_client.get_transport().is_active():
            connect_to_server()

        # 启动模型并实时接收启动信息
        startup_output = start_main_script()
        return startup_output
    except Exception as e:
        return f"无法启动模型：{e}"

# 创建Gradio界面
with gr.Blocks() as interface:
    gr.Markdown("# 知识驱动的聊天机器人")
    
    chat_history = []
    
    with gr.Row():
        with gr.Column(scale=1):
            k_hop = gr.Number(label="关联检索 k-hop", value=1)
            kownledge_used = gr.Checkbox(label="使用知识库", value=True,)
            start_button = gr.Button("启动模型")
            stop_button = gr.Button("关闭模型")
            output_text = gr.Textbox(label="模型状态", interactive=False, show_label=True)
        
        # 创建聊天记录的展示框
        chatbot = gr.Chatbot(value=chat_history, label="聊天记录", height=700, scale=15)
        
    start_button.click(fn=start_server, outputs=output_text)
    stop_button.click(fn=stop_server, outputs=output_text)

    with gr.Row():
        query_input = gr.Textbox(label="请输入您的问题", placeholder="在这里编辑消息...", show_label=False, scale=10)
        query_button = gr.Button("发送消息", scale=1)
    
    query_button.click(fn=send_query, inputs=[query_input, k_hop, kownledge_used, chatbot],
                       outputs=chatbot)
    time.sleep(1)
    query_button.click(fn=lambda: "", inputs=None, outputs=query_input)

interface.launch()
