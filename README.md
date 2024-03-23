# rag-pdf
A simple application that allows user to retrieve information from pdf.

## Usage
In windows, simply clone the repository:
```
git clone https://github.com/pasto03/rag-pdf.git
```

### Initialize agent
In __main.py__:
```python
OPENAI_API_KEY = open('openai_api_key').read()
bot = RAG("PDF_PATH", OPENAI_API_KEY)
```

### Ask question from pdf
```python
...
# generate respone, use bot.run_chain() if you want stream=False
bot.async_run_chain(
    "YOUR QUESTION", 
    model_name='gpt-3.5-turbo-16k',
    max_tokens=256,
    temperature=0
)
```

### Continuous Conversation in Terminal
```python
...
# if you want memory=False to reduce cost, use bot.chat()
chat_history, dialogue = bot.continuous_chat(
    model_name='gpt-3.5-turbo-16k', 
    max_tokens=256, 
    temperature=0
)
```

### Vector Database
In this repository, we utilized __FAISS__ as we found that FAISS performs better in retrieval of document information than other vectorstore providers such as __Pinecone__. 

However, feel free to replace the vectorstore by your own preference by simply modify the code below in __main.py__:
```python
...
# in RAG class:
self.docsearch = FAISS.from_documents(self.texts, self.embeddings)
```