import hashlib
import json
import os
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def word_counter(text: str) -> str:
    """Count words and characters in provided text."""
    return f"Words: {len(text.split())}, Characters: {len(text)}"


@tool
def generate_slug(text: str) -> str:
    """Convert text into a URL-friendly slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


@tool
def json_validator(data: str) -> str:
    """Validate whether a string contains valid JSON."""
    try:
        json.loads(data)
        return "Valid JSON"
    except json.JSONDecodeError as error:
        return f"Invalid JSON: {error}"


@tool
def text_uppercase(text: str) -> str:
    """Convert provided text into uppercase."""
    return text.upper()


@tool
def generate_sha256(text: str) -> str:
    """Generate SHA256 hash for provided text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


template = """
Answer the following questions as best you can.

You have access to these tools:

{tools}

Use this format:

Question: {input}

Thought: Think about what to do.
Action: The action to take, must be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action

(Repeat Thought/Action/Action Input/Observation if needed)

Thought: I now know the final answer.
Final Answer: Give your answer here.

IMPORTANT:
- Always follow the specified format.
- Never use Answer: instead of Final Answer:.
- Do not use Markdown headings for the final answer.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)


@st.cache_resource
def create_agent_executor(groq_api_key: str, tavily_api_key: str) -> AgentExecutor:
    search_tool = TavilySearch(tavily_api_key=tavily_api_key, max_results=3)
    tools = [
        search_tool,
        get_current_time,
        word_counter,
        generate_slug,
        json_validator,
        text_uppercase,
        generate_sha256,
    ]
    llm = ChatGroq(
        api_key=groq_api_key,
        model="openai/gpt-oss-120b",
        temperature=0,
        model_kwargs={"tool_choice": "none"},
    )
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


st.set_page_config(page_title="AI Assistant", page_icon="💬")
st.title("AI Assistant")
st.caption("Ask a question or try one of the built-in tools.")

with st.sidebar:
    st.subheader("About")
    st.write("This assistant can search the web and help with everyday text tools.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

missing_keys = [
    key_name
    for key_name in ("GROQ_API_KEY", "TAVILY_API_KEY")
    if not os.getenv(key_name)
]
if missing_keys:
    st.info(
        "Add `GROQ_API_KEY` and `TAVILY_API_KEY` to your `.env` file "
        "to start chatting."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Send a message", disabled=bool(missing_keys))
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    conversation = "\n".join(
        f'{message["role"].capitalize()}: {message["content"]}'
        for message in st.session_state.messages[:-1][-10:]
    )
    agent_input = (
        f"Conversation so far:\n{conversation}\n\nCurrent question: {user_input}"
        if conversation
        else user_input
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                executor = create_agent_executor(
                    os.environ["GROQ_API_KEY"],
                    os.environ["TAVILY_API_KEY"],
                )
                response = executor.invoke({"input": agent_input})
            except Exception as error:
                st.error(f"Could not get a response: {error}")
            else:
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
