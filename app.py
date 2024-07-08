import streamlit as st
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings as HuggingFaceEmbeddings
from kvagent import agent_supervisor
import yaml
from pydantic.json import pydantic_encoder
from json import dumps
from pprint import pprint
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.yaml import YamlOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import sys
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception


GROQ_LLM = ChatGroq(
            #model="mixtral-8x7b-32768",
            model="llama3-70b-8192",
            temperature=0,
        )

class StreamlitAssistantAnswer:
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
    def re_render_answer(self, token: str) -> None:
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)
    def confirm_answer(self, message) -> None:
        self.tokens_area.markdown(message)

class AgentConversation:
    def __init__(self, app):
        self.app = app
    async def stream_conversation(self, messages):
        assistant_answer = StreamlitAssistantAnswer()
        async for event in self.app.astream_events({"message": messages}, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    assistant_answer.re_render_answer(content)

            if kind == "on_chain_end" and event["name"] == "LangGraph":
                message = event["data"]["output"]
                logger.error("message: %s, type: %s", message, type(message))
                res = message
                match message:
                    case list():
                        res = message[0]["supervisor"]["response"]
                    case dict():
                        logger.error(message)
                        res = message["supervisor"]["response"]
                if res is not None:
                    logger.info("res: %s", res)
                    assistant_answer.confirm_answer(res)
                
                return res

kvagent = AgentConversation(app=agent_supervisor)
st.set_page_config(layout="wide")

def main():
    # Streamlit UI elements
    st.title("Experimental chatbot")

    #txt = st.text_area(
    st.markdown(
    "**[WIP] Multi agent assistant for KubeVirt**\n"
    " \nAsk me to generate Virtual Machine configurations or question about instance types and VM preferences.\n"
    " \nHere are some of the example questions:\n"
    " * What instance types are available to use?\n"
    " * Please list all available VM preferences.\n"
    " * What instance type is better to use if I want to run a virtual machine with RHEL9 image.\n"
    " * What VM preference should I choose if I want to run a windows 11 virtual machine.\n"
    " * Generate a configuration for a large virtual machine with more than 4 CPUs, meant for high performance and running fedora image."
    )


    # Input from user
    if user_input_text := st.chat_input("Let's build a VM config"):
        with st.chat_message("user"):
            st.markdown(user_input_text)
        with st.spinner("building..."):
            with st.chat_message("assistant"):
                asyncio.run(kvagent.stream_conversation({"role": "user", "content": user_input_text}))


if __name__ == "__main__":
    main()

