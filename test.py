import json
import requests
data = {
     "model": "gpt-4o", #填入想要使用的模型
     "temperature": 0.7,
     "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "讲一个有关中国改革开放的故事，100个字以内进行讲述"}],
            # "content": [
            #     {"type": "text", "text": "这张图片是什么？"},
            #     {
            #         "type": "image_url",
            #         "image_url": {
            #             "url": f"data:image/jpeg;base64,{base64_image}"  # 嵌入 Base64 图片
            #         }
            #     }
            # ]
        }
    ]
}
key = 'sk-dRTI4t9lMHu6xrRhSEU0jaM284a1RflVS4RpqCQEsmAviFU7'  #填入在网页中复制的令牌
headers = {
        'Authorization': 'Bearer {}'.format(key),
        'Content-Type': 'application/json',
    }
response = requests.request("POST", "http://123.129.219.111:3000/v1/chat/completions", headers=headers, data=json.dumps(data),timeout=300)
result = response.json()
print(result)
ans = result['choices'][0]["message"]["content"]
# print(response.status_code)
# print(response.text)
print(ans)
