import streamlit as st
import asyncio
from kvagent import vm
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
        complete_vm = message["retrieve_preference"]["virtualMachine"]

        # Convert the Pydantic object to a dictionary
        vm_dict = complete_vm.dict()

        # Convert the dictionary to a YAML string
        yaml_string = yaml.dump(vm_dict)

        # Print the YAML string
        print(yaml_string)
        
        rewriting_answer = PromptTemplate.from_template(
          "Convert the provided output into YAML.  Do not add any new fields {output}"
        )

        res_chain = rewriting_answer| GROQ_LLM| StrOutputParser()
        res = res_chain.invoke(yaml_string)
        self.tokens_area.markdown(res)

class AgentConversation:
    def __init__(self, app):
        self.app = app
    async def stream_conversation(self, messages):
        assistant_answer = StreamlitAssistantAnswer()
        async for event in self.app.astream_events({"definition": messages}, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    assistant_answer.re_render_answer(content)

            if kind == "on_chain_end" and event["name"] == "LangGraph":
                message = event["data"]["output"]
                assistant_answer.confirm_answer(message)
                
                return message

kvagent = AgentConversation(app=vm)
st.set_page_config(layout="wide")

def main():
    # Streamlit UI elements
    st.title("Experimental chatbot")

    # Input from user
    if user_input_text := st.chat_input("Let's build a VM config"):
        with st.chat_message("user"):
            st.markdown(user_input_text)
        with st.spinner("building..."):
            with st.chat_message("assistant"):
                asyncio.run(kvagent.stream_conversation({"role": "user", "content": user_input_text}))


if __name__ == "__main__":
    main()

