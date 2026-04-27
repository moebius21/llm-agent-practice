# 大模型基础实践说明文档

实践清单：
- 能用 Python 调用大模型 API
- 能用 Python 调用 Embedding API
- 能进行向量相似度计算，检索相似句
- 能本地运行 Web 聊天机器人
- 能部署到 Railway，通过公网地址访问网站


参考官方文档：
- 阿里千问大语言模型：https://bailian.console.aliyun.com/cn-beijing?spm=5176.29597918.J_tAwMEW-mKC1CPxlfy227s.1.1725133cxpCDF9&tab=api#/api/?type=model&url=3016807
- 阿里 Embedding 模型：https://bailian.console.aliyun.com/cn-beijing?spm=5176.29597918.J_tAwMEW-mKC1CPxlfy227s.1.1725133cxpCDF9&tab=api#/api/?type=model&url=2846066
- DeepSeek 大语言模型：https://api-docs.deepseek.com/zh-cn/

建议按顺序完成。每一步先跑通，再理解代码。

## 0. 准备工作

### 0.1 需要安装的软件

请提前安装、注册：

- Python 3.7 或以上版本
- PyCharm 或 VS Code
- Git
- GitHub 账号
- Railway 账号

检查 Python：

```bash
python3 --version
```

检查 Git：

```bash
git --version
```

### 0.2 需要准备的 API Key

本课程主要使用阿里云百炼-通义千问系列模型及 DeepSeek 模型。 需要准备：

```text
DASHSCOPE_API_KEY
DEEPSEEK_API_KEY
```

注意：
- API Key 不要发到微信群、QQ群、GitHub 公开仓库。
- API Key 不要写进 README。
- 部署到 Railway 时，要配置到 Railway 的 Variables 中。

## 1. 配置环境变量

### 1.1 macOS / Linux

在终端执行：

```bash
export DASHSCOPE_API_KEY="sk-你的阿里云百炼APIKey"
```

检查是否配置成功：

```bash
python3 -c "import os; print(os.getenv('DASHSCOPE_API_KEY'))"
```

如果能输出你的 Key，说明当前终端已经能读取。

注意：环境变量只对当前终端和它启动的程序有效。如果你重新打开一个终端，需要重新配置。

### 1.2 Windows PowerShell

```powershell
$env:DASHSCOPE_API_KEY="sk-你的阿里云百炼APIKey"
```

检查：

```powershell
python -c "import os; print(os.getenv('DASHSCOPE_API_KEY'))"
```

### 1.3 PyCharm 中读取不到环境变量怎么办

如果在终端中运行没问题，但 PyCharm 绿色运行按钮报错，原因通常是：

> PyCharm 不是从这个终端启动的，所以它读不到终端里的 `export`。

解决办法：

1. 打开 PyCharm 右上角运行配置。
2. 点击 `Edit Configurations...`。
3. 找到 `Environment variables`。
4. 添加：

```text
DASHSCOPE_API_KEY=sk-你的阿里云百炼APIKey
```

或者在 PyCharm 底部的 Terminal 中运行 Python 程序。

## 2. Python 调用大模型 API

### 2.1 安装依赖

```text
openai>=1.30.0
```

### 2.2 示例代码

新建 `01_chat_models.py`：

```python
import os
from openai import OpenAI


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-plus"


api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("没有读取到环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url=QWEN_BASE_URL,
)

completion = client.chat.completions.create(
    model=QWEN_CHAT_MODEL,
    messages=[
        {
            "role": "system",
            "content": "你是自然语言处理课程助教，回答要准确、简洁。",
        },
        {
            "role": "user",
            "content": "请用三句话解释什么是自然语言处理。",
        },
    ],
    temperature=0.3,
)

answer = completion.choices[0].message.content
print(answer)
```

运行：

```bash
python3 01_chat_models.py
```

运行成功后，参考 DeepSeek 官方文档，调用 DeepSeek 大模型。

继续选择任何一个文生图大模型，在代码中调用。

## 3. Python 调用 Embedding API

Embedding 模型的作用是把文本转换成向量。

大模型适合：

- 生成
- 总结
- 问答
- 改写

Embedding 模型适合：

- 相似度计算
- 语义检索
- 聚类
- 推荐
- RAG 知识库问答

阿里 Embedding 模型：https://bailian.console.aliyun.com/cn-beijing?spm=5176.29597918.J_tAwMEW-mKC1CPxlfy227s.1.1725133cxpCDF9&tab=api#/api/?type=model&url=2846066

### 3.1 实践任务说明

新建 `02_embedding_similarity.py`，调用阿里`text-embedding-v4`模型，分别获取下面四句话的句向量，然后计算两两之间的相似度。

```python
texts = [
    "我喜欢自然语言处理，尤其是大语言模型。",
    "大模型可以完成文本生成、摘要和问答任务。",
    "今天学校食堂的红烧肉很好吃。",
    "语义向量可以用来计算两个句子的相似度。",
]
```

继续，查找与“语义向量有哪些作用”这句话语义最相似的一句话。

可参考代码：
```python
import numpy as np
def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )
```

## 4. 搭建一个 Web 聊天机器人

这一部分要把大模型调用封装成一个网站。

整体流程：

```text
浏览器输入问题
  -> JavaScript 请求 /chat
  -> FastAPI 后端收到问题
  -> 后端调用大模型
  -> 返回 JSON
  -> 页面显示回答
```

可以借助 AI，建议提示词：“帮我基于阿里千问大模型搭建一个极简单网站，一个大模型chatbot。使用fastapi，后端调用大模型。代码目录结构仅包含static/index.html、app.py、requirements.txt、README.md”

### 4.1 项目结构

建议目录：

```text
customer-service-bot/
  app.py
  requirements.txt
  static/
    index.html
```

### 4.2 requirements.txt

```text
fastapi>=0.110.0
uvicorn>=0.27.0
openai>=1.30.0
```

安装：

```bash
pip3 install -r requirements.txt
```

### 实践任务说明

- 写后端脚本：app.py
- 写前端文件：static/index.html
- 本地运行网站

启动方式：

```bash
python3 -m uvicorn app:app --reload
```

浏览器打开后，测试对话效果：

```text
http://127.0.0.1:8000
```

## 5. 部署到 Railway

GitHub Pages 和 Surge.sh 只能部署静态网页，不能运行 FastAPI 后端。

我们的项目包含：

```text
HTML 前端 + FastAPI 后端 + 大模型 API 调用
```

所以需要 Railway 这类可以运行后端服务的平台。

## 5.1 Railway 部署版项目结构

可以直接复制压缩包代码。

## 5.2 上传到 GitHub

进入项目目录：

```bash
cd class2-railway
```

初始化：

```bash
git init
git add .
git commit -m "Add Railway chatbot project"
```

关联 GitHub 仓库：

```bash
git remote add origin https://github.com/你的用户名/class2-railway.git
git branch -M main
git push -u origin main
```

如果 GitHub 让你输入用户名和密码：

- 用户名：GitHub 用户名，不是邮箱
- 密码：Personal Access Token，不是 GitHub 登录密码

GitHub Token 页面：

```text
https://github.com/settings/tokens
```

## 5.3 在 Railway 部署

1. 打开 [Railway](https://railway.com/)。
2. 使用 GitHub 登录。
3. 点击 `New Project`。
4. 选择 `Deploy from GitHub repo`。
5. 选择你的 `class2-railway` 仓库。
6. 点击 `Deploy`。

第一次部署可能失败，因为还没有配置环境变量。继续下一步。

## 5.4 配置 Railway 环境变量

进入 Railway 项目中的具体服务，找到：

```text
Variables
```

添加：

```text
DASHSCOPE_API_KEY=sk-你的阿里云百炼APIKey
```

注意：

- 变量名必须完全一致：`DASHSCOPE_API_KEY`
- 不要写成 `DASHSCOPE_KEY`
- 不要写成 `DashScope_API_KEY`
- 不要加引号
- 不要把 Key 写进 GitHub

配置完变量后，重新部署。

## 5.5 生成公网域名

部署成功后，Railway 会生成一个公网地址，例如：

```text
https://class2-railway-production.up.railway.app
```

打开这个地址，就可以访问你的网站。

## 6. Railway 常见问题

### 6.1 页面能打开，但聊天提示没有读取到环境变量

错误类似：

```text
没有读取到环境变量 DASHSCOPE_API_KEY。请在 Railway Variables 中配置它。
```

排查：

1. 变量名是否完全是：

```text
DASHSCOPE_API_KEY
```

2. 配置后是否进行了重新部署。


## 7. 实践任务清单

基础任务：

- 能用 Python 调用大模型 API
- 能用 Python 调用 Embedding API
- 能进行向量相似度计算，检索相似句
- 能本地运行 Web 聊天机器人
- 能部署到 Railway，通过公网地址访问网站

提高任务：

- 优化网页样式
- 支持 DeepSeek / 千问切换
- 增加多轮对话
