#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install llama-index
# !pip install llama-cloud-services

# In[2]:


import nest_asyncio

from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from IPython.display import Image, Markdown

from llama_parse import LlamaParse

from llama_index.core.node_parser import MarkdownElementNodeParser

# In[24]:


import os

os.environ["OPENAI_API_KEY"] = "sk-XFQ2e6h_O1mz8SbGJOIYGTlaTAKXptJQLtS9aJQJYCT3BlbkFJg7Vk2Le9pn8l9gQZQrQBJw57Quz50xPgv_I9Tk64IA"

llm_o1 = OpenAI(model="gpt-4")
llm_gpt4o_mini = OpenAI(model="gpt-4o")
llm_o1_preview = OpenAI(model="gpt-3.5-turbo")

# In[4]:


parser = LlamaParse(
    api_key="llx-Ge2sxD7e9O4aR5yqmzuRzS6sXQPdmACB1k1H1N1BF7nZ66OG",
    result_type="markdown"
)

# In[5]:


documents = parser.load_data("/Users/shubhamsharma/Desktop/Exp3/Financial_Sample.xlsx")

# In[6]:


len(documents)

# In[7]:


print(documents[0].get_content())

# In[8]:


node_parser = MarkdownElementNodeParser(llm=llm_gpt4o_mini, num_workers=4)

# In[9]:


nodes = node_parser.get_nodes_from_documents(documents)

# In[10]:


base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

# In[11]:


len(nodes), len(base_nodes), len(objects)

# In[12]:


print(objects[0].get_content())

# In[13]:


# dump both indexed tables and page text into the vector index
recursive_index = VectorStoreIndex(nodes=base_nodes + objects, llm=llm_gpt4o_mini)

recursive_query_engine_o1 = recursive_index.as_query_engine(
    similarity_top_k=5, llm=llm_o1
)

recursive_query_engine_o1_preview = recursive_index.as_query_engine(
    similarity_top_k=5, llm=llm_o1_preview
)

recursive_query_engine_gpt4o_mini = recursive_index.as_query_engine(
    similarity_top_k=5, llm=llm_gpt4o_mini
)

# In[22]:


query = "During 2014, which product registered the largest drop in monthly profit from one month to the next, and which segment/country combination drove that downturn?"

response_recursive_o1 = recursive_query_engine_o1.query(query)
response_recursive_o1_preview = recursive_query_engine_o1_preview.query(query)
response_recursive_gpt4o_mini = recursive_query_engine_gpt4o_mini.query(query)

# In[23]:


print("----------------------RESPONSE WITH O1 MINI----------------------")
print(Markdown(f"{response_recursive_o1}"))

print("----------------------RESPONSE WITH O1 PREVIEW----------------------")
print(Markdown(f"{response_recursive_o1_preview}"))

print("----------------------RESPONSE WITH GPT4O-MINI----------------------")
print(Markdown(f"{response_recursive_gpt4o_mini}"))

# In[18]:


query = "Track cumulative sales month-over-month in 2014; in which month does the cumulative total cross 75 M USD, and which segment contributed most in that month?"

response_recursive_o1 = recursive_query_engine_o1.query(query)
response_recursive_o1_preview = recursive_query_engine_o1_preview.query(query)
response_recursive_gpt4o_mini = recursive_query_engine_gpt4o_mini.query(query)


# In[19]:


print("----------------------RESPONSE WITH O1 MINI----------------------")
print(Markdown(f"{response_recursive_o1}"))

print("----------------------RESPONSE WITH O1 PREVIEW----------------------")
print(Markdown(f"{response_recursive_o1_preview}"))

print("----------------------RESPONSE WITH GPT4O-MINI----------------------")
print(Markdown(f"{response_recursive_gpt4o_mini}"))

# In[ ]:


