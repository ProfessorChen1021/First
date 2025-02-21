import chromadb
from reader import read_text
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model = "mxbai-embed-large")
titlesloader = read_text("TitlesDemo.txt")
linksloader = read_text("LinksDemo.txt")
# titlesloader = TextLoader("./TitlesDemo.txt", encoding="utf-8")
# linksloader = TextLoader("./LinksDemo.txt", encoding="utf-8")
title_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
link_splitter = CharacterTextSplitter(chunk_size=70, chunk_overlap=0)

titles = title_splitter.split_text(titlesloader)
links = link_splitter.split_text(linksloader)

persistent = chromadb.PersistentClient()
collection = persistent.get_or_create_collection("original_links")
collection.add(ids = links, documents = titles)

vector_store = Chroma(
    collection_name="original_links",
    embedding_function = embeddings,
    client=persistent
)

# vector_store.add_documents(documents=titles, ids=links)