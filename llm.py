import pandas as pd
import google.generativeai as palm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
# from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.llms import VLLM
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
import gradio as gr
import requests
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
os.environ['GOOGLE_API_KEY']= KEY



print('Mapping Embeddings')
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(model_name=model_name,
     encode_kwargs=encode_kwargs)
embeddings = model_norm
# db = FAISS.from_documents(chunks,embeddings)
# db.load_local(db_path)
db_path = 'C:/Users/HP/Desktop/db_faiss'
db = FAISS.load_local(db_path,embeddings)
print('Prompt Chain')

custom_prompt_template = """You are an aersopace engineering expert your task is to review the given query and suggest possible chnages to it and refine and elaborate it so that it can provide more insightful information also quote the source of your information
You should try to identify some terms that are a bit general and broad and try to explore and elaboarte them in detail and also tell what were some of the major things missing in the original query and what negative impacts it could cause,if you are citing any information then also include a bit of context and detailed description the context of that citation and how it relates to the current query and then cite the source so that user can get a proper idea if they didn't have time to completely go through cited source and also make citations in such a way that user can understand which citation to go for a particular description
Context: {context}
Question: {question}
"""

prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



print('Creating LLM')

llm2 = GooglePalm(
           max_new_tokens=1024,
           top_k=20,
           top_p=0.3,
           temperature=0.1)
qa_chain = RetrievalQA.from_chain_type(llm=llm2,
                                      chain_type='stuff',
                                      retriever=db.as_retriever(search_kwargs={'k': 5}),
                                      return_source_documents=True,
                                      chain_type_kwargs={'prompt': prompt})
history_df = pd.DataFrame(columns = ['Question','Answer'])
def qa_bot(query):
  global history_df
  response = qa_chain({'query': query})
  print(response)
  response_df = pd.DataFrame.from_dict([response])
  print(response)
  response_df.rename(columns = {'query' : 'Question','result' : 'Answer'},inplace = True)
  print(response)
  history_df = pd.concat([history_df,response_df])
  history_df.reset_index(drop = True,inplace = True)
  print(history_df)
  return (response['result'])

logo_path = path
with gr.Blocks(theme='upsatwal/mlsc_tiet') as demo:
    title = gr.HTML("<h1>MLSCBot</h1>")
    with gr.Row():
      img = gr.Image(logo_path,label = 'MLSC Logo',show_label = False,elem_id = 'image',height = 200)
    input = gr.Textbox(label="How can I assist you?")  
    output = gr.Textbox(label="Here you go:")  
    btn = gr.Button(value="Answer",elem_classes="button-chatbot",variant = "primary")  
    btn.click(fn=qa_bot, inputs=input,outputs=output)
demo.launch(share=True, debug=True,show_api = False,show_error = False)
