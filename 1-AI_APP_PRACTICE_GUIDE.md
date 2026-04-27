# 大模型基础实践说明文档

实践清单：
- 能用 Python 调用大模型 API
- 能用 Python 调用 Embedding API
- 能进行向量相似度计算，检索相似句
- 能本地运行 Web 聊天机器人
- 能实现基于 RAG 的知识库问答机器人
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

本实践主要使用阿里云百炼-通义千问系列模型及 DeepSeek 模型。 需要准备：

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

## 5. 实践任务：基于 RAG 的知识库问答机器人

前面的 Web 聊天机器人可以调用大模型回答问题，但它还不知道企业自己的资料。比如企业套餐价格、退款规则、发票政策、售后政策等信息，如果没有提供给模型，模型可能会泛泛而谈，甚至编造答案。

本实践任务要实现一个基于 RAG 的知识库问答机器人，让模型先检索企业知识库，再基于检索到的资料回答。

RAG 的全称是 Retrieval-Augmented Generation，中文通常叫“检索增强生成”。

整体流程：

```text
企业知识库
  -> 切分成多个知识片段
  -> 调用 Embedding 模型生成向量

用户问题
  -> 调用 Embedding 模型生成问题向量
  -> 计算问题向量和知识片段向量的相似度
  -> 取最相关的几个知识片段
  -> 把知识片段和用户问题一起交给大模型
  -> 大模型基于资料生成回答
```

### 5.1 任务目标

完成一个 Python 脚本，实现：

- 读取企业知识库文件
- 将知识库切分成多个片段
- 调用 Embedding API 生成知识库向量
- 对用户问题生成向量
- 计算相似度，检索最相关的知识片段
- 将检索结果和用户问题交给大模型
- 输出最终回答

本任务先不做网站，只在终端运行，这样更容易理解 RAG 的基本原理。

### 5.2 项目结构

建议新建目录：

```text
rag-demo/
  rag_demo.py
  knowledge.txt
  requirements.txt
```

`requirements.txt`：

```text
openai>=1.30.0
numpy>=1.26.0
```

安装依赖：

```bash
pip3 install -r requirements.txt
```

### 5.3 准备企业知识库

新建 `knowledge.txt`：

```text
【公司介绍】
星河科技有限公司是一家面向中小企业提供智能客服、知识库问答和数据分析工具的软件公司。

【产品套餐】
星河智能客服目前有三个套餐。基础版价格为每月 99 元，适合个人和小团队使用。专业版价格为每月 299 元，适合成长型企业使用。企业版需要联系销售定制报价。

【发票政策】
用户付款成功后，可以在后台的“费用中心”申请电子发票。普通发票通常会在 1 个工作日内开具，专用发票通常会在 3 个工作日内开具。

【退款规则】
月付套餐购买后 7 天内，如果 AI 问答使用量少于 50 次，可以申请退款。超过 7 天或使用量达到 50 次及以上，不支持退款。

【人工客服】
如果智能客服无法解决问题，用户可以通过官网右下角的在线客服入口联系人工客服。人工客服工作时间为周一至周五 9:00-18:00。
```

注意：不同知识片段之间用空行隔开，方便程序切分。

### 5.4 核心代码思路

读取知识库：

```python
from pathlib import Path

KNOWLEDGE_FILE = Path("knowledge.txt")

def load_knowledge():
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    chunks = []
    for block in text.split("\n\n"):
        chunk = block.strip()
        if chunk:
            chunks.append(chunk)
    return chunks
```

调用 Embedding 模型：

```python
def get_embeddings(client, texts):
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=texts,
    )
    return [item.embedding for item in response.data]
```

计算相似度：

```python
import numpy as np

def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )
```

检索最相关知识片段：

```python
def search_knowledge(client, query, chunks, chunk_vectors, top_k=3):
    query_vector = get_embeddings(client, [query])[0]

    scored_chunks = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((float(score), chunk))

    scored_chunks.sort(reverse=True, key=lambda item: item[0])
    return scored_chunks[:top_k]
```

把检索结果交给大模型：

```python
def answer_with_rag(client, query, search_results):
    context_parts = []
    for index, (score, chunk) in enumerate(search_results, start=1):
        context_parts.append(f"[资料{index}，相似度 {score:.4f}]\n{chunk}")

    context = "\n\n".join(context_parts)

    system_prompt = """
你是企业智能客服。
你必须根据企业知识库资料回答用户问题。
如果资料中没有答案，请说明“知识库中暂未找到相关信息，需要人工客服进一步处理”。
回答要礼貌、简洁、专业，不要编造资料中没有的信息。
"""

    user_prompt = f"""
请根据下面的企业知识库资料回答用户问题。

企业知识库资料：
{context}

用户问题：
{query}
"""

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content
```

### 5.5 运行测试

建议测试这些问题：

```text
专业版多少钱？
```

```text
如何申请发票？
```

```text
超过 7 天还能退款吗？
```

```text
人工客服什么时候在线？
```

运行后，程序应该打印：

- 知识库片段数量
- 向量维度
- 检索到的前 3 条资料
- 大模型最终回答

### 5.6 验收要求

学生需要提交：

```text
1. rag_demo.py
2. knowledge.txt
3. 运行截图
4. 至少 3 个测试问题和回答结果
5. 简要说明 RAG 的流程
```

要求能够解释：

- 为什么要使用 Embedding
- 相似度的作用是什么
- 为什么要把检索到的资料放进 prompt
- 如果知识库没有答案，模型应该怎么回答

## 6. 部署到 Railway

GitHub Pages 和 Surge.sh 只能部署静态网页，不能运行 FastAPI 后端。

我们的项目包含：

```text
HTML 前端 + FastAPI 后端 + 大模型 API 调用
```

所以需要 Railway 这类可以运行后端服务的平台。

## 6.1 Railway 部署版项目结构

可以直接复制压缩包代码。

## 6.2 上传到 GitHub

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

## 6.3 在 Railway 部署

1. 打开 [Railway](https://railway.com/)。
2. 使用 GitHub 登录。
3. 点击 `New Project`。
4. 选择 `Deploy from GitHub repo`。
5. 选择你的 `class2-railway` 仓库。
6. 点击 `Deploy`。

第一次部署可能失败，因为还没有配置环境变量。继续下一步。

## 6.4 配置 Railway 环境变量

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

## 6.5 生成公网域名

部署成功后，Railway 会生成一个公网地址，例如：

```text
https://class2-railway-production.up.railway.app
```

打开这个地址，就可以访问你的网站。

## 7. Railway 常见问题

### 7.1 页面能打开，但聊天提示没有读取到环境变量

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


## 8. 实践任务清单

基础任务：

- 能用 Python 调用大模型 API
- 能用 Python 调用 Embedding API
- 能进行向量相似度计算，检索相似句
- 能本地运行 Web 聊天机器人
- 能实现基于 RAG 的知识库问答机器人
- 能部署到 Railway，通过公网地址访问网站

提高任务：

- 优化网页样式
- 支持 DeepSeek / 千问切换
- 增加多轮对话
- 把 RAG 知识库问答机器人接入 Web 网站
