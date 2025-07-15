# -*- coding:utf-8 -*-
# coding=utf-8
""" @Author：ndf
    @date：2025/2/19 14:43
    @desc：    
"""
import json
import re

import httpx
from icecream import ic

# 公司API

# API_URL = "http://124.237.232.34:38802/v1/chat/completions"
# MODEL_PATH = '/data2/zstax_test/eval/filco/models/DeepSeek-R1-Distill-Qwen-32B'
# API_URL = "http://124.237.232.34:38805/v1/chat/completions"
# MODEL_PATH = "/app/DeepSeek-R1-Distill-Qwen-7B"

# 北京税局API
API_URL = "http://192.168.5.50:8000/v1/chat/completions"
MODEL_PATH = "/data/servers/QwQ-32B-GGUF/qwq-32b-q8_0.gguf"

class ChatSession:
    def __init__(self, history, model=MODEL_PATH, max_history=5):
        self.api_url = API_URL
        self.model = model
        self.history = history
        self.max_history = max_history

    def chat(self, user_input):
        # 维护历史记录，限制对话轮数
        if len(self.history) > self.max_history*2:
            self.history = self.history[-self.max_history*2:]

        # 追加用户输入
        self.history.append({"role": "user", "content": user_input})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f""
        }
        data = {
            "model": self.model,
            "messages": self.history,
            "temperature": 0.01,
            "top_p": 0.1,
        }
        try:
            r = httpx.post(self.api_url, json=data, headers=headers, timeout=None)
            # 解析回复
            result = r.json()
            assistant_reply = result["choices"][0]["message"]["content"]
            # 追加 AI 回复
            self.history.append({"role": "assistant", "content": assistant_reply})
            print(self.history)
            return assistant_reply
        except Exception as e:
            ic(e)


def request_deepseek(query):
    messages = [
        {'role': 'user', 'content': query}
    ]
    url = API_URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f""
    }
    data = {
        "model": MODEL_PATH,
        "messages": messages,
        "temperature": 0.01,
        "top_p": 0.1,
    }

    try:
        r = httpx.post(url, json=data, headers=headers, timeout=None)
        return r.json()
    except Exception as e:
        ic(e)

def get_json_rlt(query):
    res = request_deepseek(query)
    text = (res["choices"][0]["message"]["content"])
    # 使用正则表达式过滤掉 <think> 标签及其内部内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned_text=cleaned_text.replace('```json', '').replace('```', '').strip()
    return json.loads(cleaned_text)

if __name__ == '__main__':
    res = request_deepseek("我上一次问的是什么？")
    res_json = (res["choices"][0]["message"]["content"])
    print(res_json.strip())
    # chat = ChatSession([])
    # print(chat.chat("你好呀"))
    # print("---------------")
    # print(chat.chat("你是谁"))
    # print("---------------")
    # print(chat.chat("我的上一个问题是什么"))
