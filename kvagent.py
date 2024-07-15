
# New iteration of an attempt to build VMs generating agent

import os
from pprint import pprint
import jq
import uuid
import lark
import asyncio
import yaml
from pydantic.json import pydantic_encoder
from json import dumps

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import LLMChain
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.yaml import YamlOutputParser

import chromadb.utils.embedding_functions as embedding_functions

from typing import List, Optional, Dict

import chromadb
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from kvtypes import *
from kvtypes import RelatedInstanceTypes, CallAgent, DataVolumeTemplateSpec, Volume
from vmpreferences import prefs_document_content_description, prefs_metadata_field_info
from vminstancetypes import instTypes_metadata_field_info, instTypes_document_content_description
from bootableSources import find_bootable_sources, bootSrcs_document_content_description, bootSrcs_metadata_field_info

import sys
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


model_name = "BAAI/bge-base-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

#bge_embeddings = HuggingFaceBgeEmbeddings(
#    model_name=model_name,
#    model_kwargs={'device': 'cuda'},
#    encode_kwargs=encode_kwargs
#)


embedding_model=model_name

#embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model, device="cuda", normalize_embeddings=True)
#e = SentenceTransformerEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cuda'}, encode_kwargs=encode_kwargs)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model, normalize_embeddings=True)
e = SentenceTransformerEmbeddings(model_name=embedding_model, encode_kwargs=encode_kwargs)

persist_directory = 'db'

client = chromadb.PersistentClient(path=persist_directory)

# we will need two collections. One for instance types and another for preferences
collectionInstTypes = client.get_or_create_collection("instanceTypes",
                                      embedding_function=embedding_func)

collectionPref = client.get_or_create_collection("prefs",
                                      embedding_function=embedding_func)

collectionBootSources = client.get_or_create_collection("bootSrcs",
                                      embedding_function=embedding_func)

def load_bootable_sources():
    docs = []
    srcs = find_bootable_sources()
    for doc in srcs:
        docs.append(Document(
            page_content=doc['description'],
            metadata={"name": doc['name']}))
    return docs

def fix_metadata(original_metadata):
    new_metadata = {}
    for k, v in original_metadata.items():
        if type(v) in [str, int, float]:
           # str, int, float are the types chroma can handle
            new_metadata[k] = v
        elif isinstance(v, list):
            new_metadata[k] = ','.join(v)
        else:
            # e.g. None, bool
            new_metadata[k] = str(v)
    return new_metadata

def split_json_docs(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    for doc in docs:
        doc.metadata=doc.metadata
    return docs


def metadata_func(record: dict, metadata: dict) -> dict:
    record = record['metadata']
    for key, val in record.items():
        metadata[key] = val
    return metadata

def loadCollection(collection, documents):
    names = set()
    split_json_documents = split_json_docs(documents)
    for doc in split_json_documents:
        if doc.metadata['name'] not in names:
            collection.add(
                ids=[str(uuid.uuid1())],
                metadatas=doc.metadata,
                documents=doc.page_content
            )
            names.add(doc.metadata['name'])


if collectionInstTypes.count() < 1:
    instanceTypesFile = "./formattedInstTypesCollection.json"
    loader = JSONLoader(file_path=instanceTypesFile, jq_schema=".VirtualMachineInstancetypes[]", content_key="description", metadata_func=metadata_func) ## text_content=False)
    documents = loader.load()
    loadCollection(collectionInstTypes, documents)


if collectionPref.count() < 1:
    instancePrefsFile = "./formattedPrefCollection.json"
    loaderPrefs = JSONLoader(file_path=instancePrefsFile, jq_schema=".VirtualMachinePreferences[]", content_key="description", metadata_func=metadata_func) ## text_content=False)
    prefDocs = loaderPrefs.load()
    loadCollection(collectionPref, prefDocs)

# load boot sources
loadCollection(collectionBootSources, load_bootable_sources())



# Load LLM


#llama3-70b-8192
#"llama2-70b-4096"
#"gemma-7b-it"
GROQ_LLM = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
        )
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


vectordbInstTypes = Chroma(persist_directory=persist_directory, embedding_function=e, collection_name = 'instanceTypes')
vectordbPrefs = Chroma(persist_directory=persist_directory, embedding_function=e, collection_name = 'prefs')
vectordbBootSrcs = Chroma(persist_directory=persist_directory, embedding_function=e, collection_name = 'bootSrcs')



retrieverInstTypes = SelfQueryRetriever.from_llm(
    llm,
    vectordbInstTypes,
    instTypes_document_content_description,
    instTypes_metadata_field_info,
    verbose=True,
    structured_query_translator = ChromaTranslator(),
    fix_invalid=True,

)

retrieverPrefs = SelfQueryRetriever.from_llm(
    llm,
    vectordbPrefs,
    prefs_document_content_description,
    prefs_metadata_field_info,
    verbose=True,
    structured_query_translator = ChromaTranslator(),
    fix_invalid=True,
)

retrieverBootSources = SelfQueryRetriever.from_llm(
    llm,
    vectordbBootSrcs,
    bootSrcs_document_content_description,
    bootSrcs_metadata_field_info,
    verbose=True,
    structured_query_translator = ChromaTranslator(),
    fix_invalid=True,
)


#Build the vector retrievers

gen_instances_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You need to select a single most suitable instance type from the provided context. Your answer should be based on the user query which contains the requirements for defining a virtual machine. Here is the context: {context}. If you don't know the answer, just say that you don't know.""",
        ),
        ("user", "Query: {query}"),
    ]
)

gen_instTypes_chain = gen_instances_prompt | llm.with_structured_output(RelatedInstanceTypes)


gen_preferences_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You need to select a single most suitable virtual machine preference from the provided context. Your answer should be based on the user query which contains the requirements for defining a virtual machine. Here is the context: {context}. If you don't know the answer, just say that you don't know.""",
        ),
        ("user", "Query: {query}"),
    ]
)

gen_preferences_chain = gen_preferences_prompt | llm.with_structured_output(RelatedPreferences)



#Generate initial Virtual Machine draft

gen_vm_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a virtual machine configuration writer. Write a configuration for a virtual machine based on a user-provided request. Be very specific.",
        ),
        ("user", "{request}"),
    ]
)

generate_vm = gen_vm_prompt | GROQ_LLM.with_structured_output(
    VirtualMachine
)

async def generate_draft_vm(state: VmCreationState):
    definition = state["definition"]

    # Query rewriting prompt
    #rewriting_prompt = PromptTemplate.from_template(
    #  "Rewrite the following query by extracting the specific requirements for defining a virtual machine from the use query. Use only the words from the use query. Do not add any new requirements: {query}"
    #)

    # LLM Chain for query rewriting
    #rewriting_chain = rewriting_prompt| GROQ_LLM| StrOutputParser()

    # Run the chain
    #definition = await rewriting_chain.ainvoke({"query": definition})
    results = await generate_vm.ainvoke({"request": definition})
    
    return {
        **state,
        "definition": definition,
        "virtualMachine": results,
    }

async def retrieve_instance_type(state: VmCreationState):
    definition = state["definition"]
    vmdef = state["virtualMachine"]
    try:
        CONTEXT = await retrieverInstTypes.ainvoke(definition)
    except:
        CONTEXT=[]
    if len(CONTEXT) == 0:
        retriever_from_llm = MultiQueryRetriever.from_llm(vectordbInstTypes.as_retriever(), llm=llm)
        CONTEXT = await retriever_from_llm.ainvoke(definition)
        
    result = await gen_instTypes_chain.ainvoke({"query": definition, "context":CONTEXT})

    if result is None:
        result = {"instanceTypes": [{"name": "no instance type"}]}

    vmdef.spec.instancetype.name = result.instanceTypes[0].name
    return {
        **state,
        "virtualMachine": vmdef,
    }

  
async def retrieve_preference(state: VmCreationState):
    definition = state["definition"]
    vmdef = state["virtualMachine"]

    try:
        CONTEXT = await retrieverPrefs.ainvoke(definition)
    except:
        CONTEXT=[]
    if len(CONTEXT) == 0:
        retriever_from_llm = MultiQueryRetriever.from_llm(vectordbPrefs.as_retriever(), llm=llm)
        CONTEXT = await retriever_from_llm.ainvoke(definition)
    
    result = await gen_preferences_chain.ainvoke({"query": definition, "context":CONTEXT})
    if result is None:
        result = {"preferences": [{"name": "no preference"}]}

    vmdef.spec.preference.name = result.preferences[0].name
    return {
        **state,
        "virtualMachine": vmdef,
        "complited": True,
    }

async def finalize(state: VmCreationState):
    if state["complited"]:
        return END
    return "init_vm"

async def handle_volumes(state: VmCreationState):
    definition = state["definition"]
    vm = state["virtualMachine"]

    # Query rewriting prompt to find out which OS should be used for boot
    os_query_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your goal is to understand what operating system should be used to boot the virtual machine the user is requesting in the user query. If the user didn't explicitly mentoin the operating system, please assume that it's Fedora. Then, compose a qestion for a vector store retriever to explicitly retrieve the desired operating system.",
            ),
            ("user", "Query: {query}"),
        ]
    )

    # LLM Chain for query rewriting
    rewriting_chain = os_query_prompt| GROQ_LLM| StrOutputParser()
    # Run the chain
    definition = await rewriting_chain.ainvoke({"query": definition})

    try:
        CONTEXT = await retrieverBootSources.ainvoke(definition)
    except:
        CONTEXT=[]
    if len(CONTEXT) == 0:
        retriever_from_llm = MultiQueryRetriever.from_llm(vectordbBootSrcs.as_retriever(), llm=llm)
        CONTEXT = await retriever_from_llm.ainvoke(definition)

    if len(CONTEXT) > 0:

        context_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Use the provided context {context} to select only oned atavolume source, most suitable for the requestion operating system in the user query. if the user didn't request any specific operating system, assume fedora is most suitable.",
                ),
                ("user", "Query: {query}"),
            ]
        )
        
        gen_boot_source_chain = context_prompt | llm.with_structured_output(RelatedBootSource)
        result = await gen_boot_source_chain.ainvoke({"query": definition, "context":CONTEXT})
        if result:

            # ---- rewrite the volume section
            rewrite_volumes_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Generate a data volume template based on the provided context: {context}. The datavolume name must be the name provided in the context.",
                    ),
                ]
            )
            
            gen_boot_source_chain = rewrite_volumes_prompt | llm.with_structured_output(DataVolumeTemplateSpec)
            datavolume =  await gen_boot_source_chain.ainvoke({"context": result.bootsources[0].name})
            vm.spec.dataVolumeTemplates = [datavolume]
            # Convert the Pydantic object to a dictionary
            if isinstance(vm, dict):
                vm_dict = vm
            else:
                vm_dict = vm.dict()
            # Convert the vol dictionary to a json string
            json_vol_string = dumps(vm_dict['spec']['template']['spec']['volumes'])
            vmspec = dumps(vm_dict)
            vol_context_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Rewrite the provided volumes section {volumes} based on the requrements provided by the user query. One of the volumes must be a dataVolume. The dataVolume name must exactly match the provided {dname}. The number of volumes must be equal to the number of disks and the name of the volume must be identical to the name of the corresponding disk from the context {context}",
                    ),
                    ("user", "Query: {query}"),
                ]
            )
            gen_vol_chain = vol_context_prompt | llm.with_structured_output(RelatedVolumes)
            volumes =  await gen_vol_chain.ainvoke({"dname": datavolume.metadata.name, 'volumes': json_vol_string, "query": definition, "context": vmspec})
            try: 
                for idx in range(len(volumes.volumes)):
                    if volumes.volumes[idx].dataVolume:
                        volumes.volumes[idx].dataVolume.name = datavolume.metadata.name
            except Exception as e:
                print("ERROR: %s" % e)
            vm.spec.template.spec.volumes = volumes.volumes
    return {
        **state,
        "virtualMachine": vm,
    }

vm_builder = StateGraph(VmCreationState)

nodes = [
    ("init_vm", generate_draft_vm),
    ("retrieve_instance_type", retrieve_instance_type),
    ("retrieve_preference", retrieve_preference),
    ("handle_volumes", handle_volumes),
]
for i in range(len(nodes)):
    name, node = nodes[i]
    vm_builder.add_node(name, node)
    if i > 0:
        vm_builder.add_edge(nodes[i - 1][0], name)

vm_builder.add_conditional_edges("retrieve_preference", finalize)

vm_builder.set_entry_point(nodes[0][0])
vm = vm_builder.compile(debug=True).with_config(run_name="Construct Virtual Machine configuration")


class AssistantState(TypedDict):
    message: str 
    response: str
    callAgent: str

async def get_supervisor_response(state: AssistantState):
    if state["response"]:
        return {
            **state,
            "callAgent": "END",
        }
    system_prompt = (
        "You are a supervisor tasked to choose which agent to run between"
        " following workers:  {members}. Evaluate the following user request,"
        " you should respond with VMBuilder only if the user explicitly requests to build, compose, generate or construct a configuration for a Virtual Machine."
        " You must return InstanceTypeLookup if the user is asking about virtual machine instance types or simply instance types. InstanceTypeLookup is not a tool, but a name of a member."
        " For questions about preferences or virtual machine preferences you should respond with PreferencesLookup. PreferencesLookup is not a tool, but a name of a member." 
        " If the user is inquiring boot sources, operating systems to boot from or images that can be used for booting a virtual machine, you should respond with BootSourceLookup. BootSourceLookup is not a tool or a function, but a name of a member." 
        " For any other questions you should respond with LLM."
    )
    query = state["message"]
    members = ["VMBuilder", "InstanceTypeLookup", "PreferencesLookup", "BootSourceLookup", "LLM"]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Query: {query}"),
            (
                "system",
                "Given the user query above, who should act next?"
                "Select one of: {options}",
            ),
        ]
    )
    gen_callagent_chain = prompt | GROQ_LLM.with_structured_output(CallAgent)
    result = await gen_callagent_chain.ainvoke({"query": query, "members":", ".join(members), "options": str(members)})

    return {
        **state,
        "callAgent": result.callAgent,
    }

async def get_response_from_llm(state: AssistantState):
    query = state["message"]

    # Query rewriting prompt
    prompt = PromptTemplate.from_template(
      "You are a helpfull assistant, please answer the user query the best you can: {query}"
    )

    # LLM Chain to answer use query
    chain = prompt| GROQ_LLM| StrOutputParser()

    # Run the chain
    result = await chain.ainvoke({"query": query})
    
    return {
        **state,
        "response": result,
    }

async def lookup_instance_types(state: AssistantState):
    query = state["message"]
    try:
        CONTEXT = await retrieverInstTypes.ainvoke(query, search_kwargs={"k":100})
    except:
        CONTEXT=[]
    try:
        retriever_from_llm = MultiQueryRetriever.from_llm(vectordbInstTypes.as_retriever(search_kwargs={"k": 100}), llm=llm)
        CONTEXT1 = await retriever_from_llm.ainvoke(query)
    except Exception as e:
        CONTEXT1 = []
    CONTEXT=CONTEXT+CONTEXT1
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert on virtual machine instance types. you need to help answering questions about instance type from the provided context. Your answer should be based on the user query which contains the requirements.  Here is the context: {context}. If you don't know the answer, just say that you don't know.""",
            ),
            ("user", "Query: {query}"),
        ]
    )
    chain = prompt| GROQ_LLM| StrOutputParser()

    # Run the chain
    result = await chain.ainvoke({"query": query, "context": CONTEXT})
        
    return {
        **state,
        "response": result,
    }

  
async def lookup_preferences(state: AssistantState):
    query = state["message"]

    try:
        CONTEXT = await retrieverPrefs.ainvoke(query)
    except:
        CONTEXT=[]
    try:
        retriever_from_llm = MultiQueryRetriever.from_llm(vectordbPrefs.as_retriever(search_kwargs={"k": 100}), llm=llm)
        CONTEXT1 = await retriever_from_llm.ainvoke(query)
    except Exception as e:
        CONTEXT1 = []
    CONTEXT=CONTEXT+CONTEXT1

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert on virtual machine preferences. You need to help answering questions about these preferences from the provided context. Your answer should be based on the user query which contains the requirements. Here is the context: {context}. If you don't know the answer, just say that you don't know.""",
            ),
            ("user", "Query: {query}"),
        ]
    )
    
    chain = prompt| GROQ_LLM| StrOutputParser()

    # Run the chain
    result = await chain.ainvoke({"query": query, "context": CONTEXT})
    return {
        **state,
        "response": result,
    }

async def lookup_bootsources(state: AssistantState):
    query = state["message"]

    try:
        CONTEXT = await retrieverBootSources.ainvoke(query)
    except:
        CONTEXT=[]
    try:
        retriever_from_llm = MultiQueryRetriever.from_llm(vectordbBootSrcs.as_retriever(search_kwargs={"k": 100}), llm=llm)
        CONTEXT1 = await retriever_from_llm.ainvoke(query)
    except Exception as e:
        CONTEXT1 = []
    CONTEXT=CONTEXT+CONTEXT1

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You need to help answering questions about available operating systems that can be used as a boot source from the provided context. Your answer should be based on the user query which contains the requirements. Here is the context: {context}. If you don't know the answer, just say that you don't know.""",
            ),
            ("user", "Query: {query}"),
        ]
    )
    
    chain = prompt| GROQ_LLM| StrOutputParser()

    # Run the chain
    result = await chain.ainvoke({"query": query, "context": CONTEXT})
    return {
        **state,
        "response": result,
    }

async def build_vm_config(astate: AssistantState):
    query = astate["message"]
    # Run the chain
    try: 
        result = await vm.ainvoke({"definition": query})
    except Exception as e:
        result = {"retrieve_preference": {"virtualMachine": {}}}
    logger.info("result :")
    logger.info(result)
    try:
        complete_vm = result["__end__"]["virtualMachine"]
    except:
        try:
            complete_vm = result["virtualMachine"]
        except:
            complete_vm = {}

    # Convert the Pydantic object to a dictionary
    if isinstance(complete_vm, dict):
        vm_dict = complete_vm
    else:
        vm_dict = complete_vm.dict()

    # Convert the dictionary to a YAML string
    yaml_string = yaml.dump(vm_dict)

    # Print the YAML string
    rewriting_answer = PromptTemplate.from_template(
      "Convert the provided output into YAML.  Do not add any new fields {output}"
    )

    res_chain = rewriting_answer| GROQ_LLM| StrOutputParser()
    res = await res_chain.ainvoke(yaml_string)
    return {
        **astate,
        "response": res,
    }


supervisor_builder = StateGraph(AssistantState)
supervisor_builder.add_node("supervisor", get_supervisor_response)
supervisor_builder.add_node("VMBuilder", build_vm_config)
supervisor_builder.add_node("PreferencesLookup", lookup_preferences)
supervisor_builder.add_node("InstanceTypeLookup", lookup_instance_types)
supervisor_builder.add_node("BootSourceLookup", lookup_bootsources)
supervisor_builder.add_node("LLM", get_response_from_llm)

supervisor_builder.add_edge("VMBuilder", "supervisor")
supervisor_builder.add_edge("PreferencesLookup", "supervisor")
supervisor_builder.add_edge("InstanceTypeLookup", "supervisor")
supervisor_builder.add_edge("BootSourceLookup", "supervisor")
supervisor_builder.add_edge("LLM", "supervisor")

supervisor_builder.add_conditional_edges(
    "supervisor",
    lambda x: x["callAgent"],
    {"VMBuilder": "VMBuilder",
    "InstanceTypeLookup": "InstanceTypeLookup", 
    "PreferencesLookup": "PreferencesLookup",
    "BootSourceLookup": "BootSourceLookup",
    "END": END,
    "LLM": "LLM"},
)


supervisor_builder.set_entry_point("supervisor")
agent_supervisor = supervisor_builder.compile(debug=True)

