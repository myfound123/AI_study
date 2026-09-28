# 导入刚才安装的 ollama 库
import ollama

print("正在呼叫本地 AI，请稍候...\n")

# 向本地模型发送消息
response = ollama.chat(
    model='qwen2.5:7b', # 这里必须和你刚才下载的模型名完全一致
    messages=[
        {
            'role': 'user',
            'content': '作为一个本地大模型你能帮我做哪些事？通过分类的方法排列能力强弱。相比于云端大模型你的能力强在哪里你的弱点又在哪里',
        },
    ]
)

# 打印 AI 的回复
print("【AI 回复】：")
print(response['message']['content'])