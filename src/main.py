import chromadb

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from langchain.tools import Tool

import streamlit as st
import random as rd
import time

llm = ChatOpenAI(
    temperature=0,
    api_key="",
    model="gpt-4o-2024-11-20"
)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("pharma-kb")

def search_pharma_kb(query, top_n=10):
    results = collection.query(
        query_texts=[query],
        n_results=top_n
    )

    return {"documents": results["documents"]}


search_pharma_kb_tool = Tool(
    name="pharma_kb_search",
    func=search_pharma_kb,
    description="Pharamaceutical knowledge base"
)

qa_agent = create_react_agent(
    model=llm,
    tools=[search_pharma_kb_tool],
    name="question-answer_expert",
    prompt="You are a pharmaceutical expert specializing in answering user questions about medication, composition, dosage, and side effects. Use the pharma knowledge base to retrieve relevant information and answer the user's questions in a detailed, natural language format. Provide references from the dataset when necessary. Just use the chunks from the knowledge to answer."
)

recommender_agent = create_react_agent(
    model=llm,
    tools=[search_pharma_kb_tool],
    name="expert-recommender",
    prompt="You are a pharmaceutical expert in recommending medications. Based on user symptoms or conditions, recommend the appropriate medication while warning about possible harmful combinations or contraindications. Use the pharma knowledge base to retrieve relevant details and ensure your recommendations are safe and personalized. Just use the chunks from the knowledge to answer."
)

alternatives_agent = create_react_agent(
    model=llm,
    tools=[search_pharma_kb_tool],
    name="alternatives-generator",
    prompt="You are an expert in pharmaceutical alternatives. Based on user queries, suggest safe alternatives for medications. Use the pharma knowledge base to ensure the alternatives are appropriate and explain any differences in effects, risks, and usage instructions. Just use the chunks from the knowledge to answer."
)

summarizer_agent = create_react_agent(
    model=llm,
    tools=[search_pharma_kb_tool],
    name="product-summarizer",
    prompt="You are an expert at summarizing pharmaceutical products. Provide a concise and clear summary of the product's key details, including its purpose, administration instructions, side effects, and warnings. Use the pharma knowledge base for accurate information. Just use the chunks from the knowledge to answer."
)

workflow = create_supervisor(
    [qa_agent, recommender_agent, alternatives_agent, summarizer_agent],
    model=llm,
    prompt=(
        """You are a supervisor managing four agents: a question-answer expert, a recommender, an alternatives generator, and a product summarizer. Based on the user's query, invoke the appropriate agent to retrieve the required information and provide detailed, well-structured answers or recommendations. 
        Ensure that:
        1. The response is concise and clearly structured.
        2. Unnecessary punctuation, such as excessive colons, dashes, or other non-essential symbols, is avoided.
        3. The information is easy to read and free of clutter.
        4. Provide the final answer in a well-formatted manner using bullet points, sections, or headings where necessary."""
    )
)

def response_generator(prompt):
    app = workflow.compile()
    result = app.invoke({
        "messages":[ {
            "role":"user",
            "content": prompt,
            }
        ]
    })

    response_content = result["messages"][-1].content
    return response_content

def main():
    st.title("Pharma Assistant💊")
    if "messages" not in st.session_state:
        st.session_state.messages = list()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt:= st.chat_input("Ask your pharmaceutical-related question"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("assistant"):
            response = response_generator(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

if __name__ == "__main__":
    main()