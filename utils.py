from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import langchain_core
from langchain_core.callbacks import BaseCallbackHandler


# 1. load and chunk pdf
def load_pdf(fpath):
    loader = PyPDFLoader(fpath)

    data = loader.load()
    return data

def chunk_data(loader_data, chunk_size=2000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_documents(loader_data)
    return texts

def remove_doc_dups(docs: list[langchain_core.documents.base.Document]):
    exist_contents = []
    new_docs = []
    for doc in docs:
        if not doc.page_content in exist_contents:
            exist_contents.append(doc.page_content)
            new_docs.append(doc)
    return new_docs


class StreamCallbackHandler(BaseCallbackHandler):
    def on_text(self, text, *args, **kwargs):
        # Process each text response here
        print(text)

