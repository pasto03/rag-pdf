from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

from langchain_core.callbacks import StreamingStdOutCallbackHandler

import os
import tiktoken

from langchain_core.messages import HumanMessage

from utils import *


class RAG:
    def __init__(self, pdf_path, OPENAI_API_KEY, index_name, chunk_size=1024, **kwargs):
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.index_name = index_name

        # load and chunk pdf
        self.texts = load_pdf(pdf_path)
        print("Raw pdf file loaded.")
        self.texts = chunk_data(self.texts, chunk_size=chunk_size)
        self.docsearch = FAISS.from_documents(self.texts, self.embeddings)

        print("Vector database built successfully.")

        # setup default chain_func for chat function
        self.chain_func = self.async_run_chain
    
    @staticmethod
    def get_num_tokens(string: str, model_name="gpt-3.5-turbo-16k"):
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(string))
        
    def summarize_dialogue(self, dialogue: str, model_name="gpt-3.5-turbo-16k", 
                           summarize_tokens=2048, max_tokens=1526, temperature=0):
        """
        automatically summarize diagolue dialogue is too long
        """
        if self.get_num_tokens(dialogue, model_name) < summarize_tokens:
            return dialogue
        llm = ChatOpenAI(temperature=temperature, openai_api_key=self.OPENAI_API_KEY, 
                         model_name="gpt-3.5-turbo-16k", max_tokens=max_tokens)
        config = "Please help me summarize the dialogue between the two characters in one sentence each."
        messages = [HumanMessage(content=config+"\n"+dialogue)]
        output = llm(messages).content
        return output

    def run_chain(self, query=None, model_name='gpt-3.5-turbo-16k', max_tokens=256, temperature=0):
        """
        run chain
        """
        llm = ChatOpenAI(temperature=temperature, openai_api_key=self.OPENAI_API_KEY, model_name=model_name, 
                         max_tokens=max_tokens)
        chain = load_qa_chain(llm, chain_type="stuff")
        if not query:
            query = input("Enter your question:")
        docs = remove_doc_dups(self.docsearch.similarity_search(query))
        output = chain.run(input_documents=docs, question=query)
        return output
    
    def async_run_chain(self, query=None, model_name='gpt-3.5-turbo-16k', max_tokens=256, temperature=0):
        """
        asynchronous chain
        """
        llm = ChatOpenAI(temperature=temperature, openai_api_key=self.OPENAI_API_KEY, model_name=model_name, 
                        streaming=True, callbacks=[StreamingStdOutCallbackHandler()], max_tokens=max_tokens)
        chain = load_qa_chain(llm, chain_type="stuff")
        if not query:
            query = input("Enter your question:")
        docs = remove_doc_dups(self.docsearch.similarity_search(query))
        callbacks = [StreamCallbackHandler()]
        output = chain.run(input_documents=docs, question=query, callbacks=callbacks)
        return output
    
    def chat(self, model_name='gpt-3.5-turbo-16k', max_tokens=256, temperature=0):
        """
        chat with bot where conversation is not continuous
        """
        user_input = ""
#         chat_text = ""
        chat_history = {"model": model_name, "messages": [], 
                        "temperature": temperature, "max_tokens": max_tokens, "continuous": False}
        
        while True:
            user_input = input("User  :")
            if user_input == "q":
                break
            else:
                chat_history['messages'].append({"role": "user", "content": user_input})
                print("Bot  :", end="")
                response = bot.chain_func(query=user_input, max_tokens=max_tokens)
                print("\n")
                chat_history['messages'].append({"role": "assistant", "content": response})
        return chat_history
    
    def continuous_chat(self, model_name='gpt-3.5-turbo-16k', max_tokens=256, temperature=0):
        """
        chat with bot where conversation is continuous
        """
        user_input = ""
        dialogue = ""
        chat_history = {"model": 'gpt-3.5-turbo-16k', "messages": [], 
                        "temperature": temperature, "max_tokens": max_tokens, "continuous": True}
        
        while True:
            user_input = input("User  :")
            if user_input == "q":
                break
            else:
                chat_history['messages'].append({"role": "user", "content": user_input})
                print("Bot  :", end="")
                dialogue += "User: " + user_input + "\n"
                dialogue = self.summarize_dialogue(dialogue, model_name)
                
                response = bot.chain_func(query=dialogue, max_tokens=max_tokens)
                print("\n")
                chat_history['messages'].append({"role": "assistant", "content": response})
                dialogue += "Assistant: " + response + "\n\n"
        return chat_history, dialogue
    

if __name__ == "__main__":
    OPENAI_API_KEY = open('openai_api_key').read()
    index_name = 'animals'
    bot = RAG("./1707.06347.pdf", OPENAI_API_KEY, index_name)

    # chat with bot with dialogue memory
    # chat_history, dialogue = bot.continuous_chat(max_tokens=512)

    # ask one question and get asnyc response
    bot.async_run_chain("summarize \"background: policy optimization\" part of this paper.", max_tokens=512)
