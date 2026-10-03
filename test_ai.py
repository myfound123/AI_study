# 导入刚才安装的 ollama 库
import ollama

print("正在呼叫本地 AI，请稍候...\n")

# 向本地模型发送消息
response = ollama.chat(
    model='qwen2.5:7b', # 这里必须和你刚才下载的模型名完全一致
    messages=[
        {
            'role': 'user',
            'content': '作为本地大模型，你能不能短暂的记住上下文，如果可以你能不能回答上次问你的是什么问题；如果不可以，帮我写一段在vscode上运行的向你提问的且可以记住上下文的代码',
        },
    ]
)

# 打印 AI 的回复
print("【AI 回复】：")
print(response['message']['content'])