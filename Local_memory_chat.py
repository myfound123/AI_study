import ollama

def chat_with_memory():
    print("🤖 本地千问助手已上线（输入 '退出' 结束对话）")
    print("-" * 40)
    
    # 初始化聊天记录，系统提示词可以设定它的角色
    messages = [
        {"role": "system", "content": "你是一位精通Python和OpenCV的计算机视觉专家，正在辅导一名大一园艺学生。请用大白话解释代码，多用农业或生活类比。"}
    ]

    while True:
        user_input = input("\n👤 你：")
        if user_input in ['退出', 'exit', 'quit']:
            print("👋 再见！")
            break
            
        # 把用户的问题加入历史记录
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 发送给本地模型时，携带所有的 messages 历史
            response = ollama.chat(model='qwen2.5:7b', messages=messages)
            reply = response['message']['content']
            
            # 打印回答，并把 AI 的回答也加入历史记录，形成闭环
            print(f"\n🤖 AI：{reply}")
            messages.append({"role": "assistant", "content": reply})
            
            # 安全机制：防止历史记录过长导致爆显存/内存
            if len(messages) > 10:
                # 保留系统提示词和最近9条对话
                messages = [messages[0]] + messages[-9:]
                
        except Exception as e:
            print(f"调用出错：{e}")

if __name__ == "__main__":
    chat_with_memory()
