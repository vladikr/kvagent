
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
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import LLMChain
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.yaml import YamlOutputParser

import chromadb.utils.embedding_functions as embedding_functions
import chromadb
from langchain.vectorstores import Chroma

from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from kvtypes import *
from kvtypes import RelatedInstanceTypes

model_name = "BAAI/bge-base-en"
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
        doc.metadata=fix_metadata(doc.metadata)
    return docs


def metadata_func(record: dict, metadata: dict) -> dict:

    for key, val in record.items():
        metadata[key] = val
    return metadata

def loadCollection(collection, documents):
  split_json_documents = split_json_docs(documents)
  for doc in split_json_documents:
      collection.add(
          ids=[str(uuid.uuid1())],
          metadatas=doc.metadata,
          documents=doc.page_content
      )


if collectionInstTypes.count() < 1:
  instanceTypesFile = "./formattedInstTypesCollection.json"
  loader = JSONLoader(file_path=instanceTypesFile, jq_schema=".VirtualMachineInstancetypes[]", content_key="descritpion", metadata_func=metadata_func) ## text_content=False)
  documents = loader.load()
  loadCollection(collectionInstTypes, documents)

if collectionPref.count() < 1:
  instancePrefsFile = "./formattedPrefCollection.json"
  loaderPrefs = JSONLoader(file_path=instancePrefsFile, jq_schema=".VirtualMachinePreferences[]", content_key="descritpion", metadata_func=metadata_func) ## text_content=False)
  prefDocs = loaderPrefs.load()
  loadCollection(collectionPref, prefDocs)


instTypes_document_content_description = "desciption of the virtual machine instance types"

instTypes_metadata_field_info = [
    AttributeInfo(
        name="cpu",
        description="Specifies the number of virtual CPUs the virtual machines requires",
        type="integer",
    ),
        AttributeInfo(
        name="dedicatedCPUPlacement",
        description="Indicates whether the virtual machines requires dedicated CPUs. dedicated CPUs also means excluse non-shared cpus. This is needed for higher performance",
        type="bool",
    ),
        AttributeInfo(
        name="isolateEmulatorThread",
        description="Indicates whether the virtual machines requires its emulator thread to be isolated on separate physcial CPU. This is needed for higher performance",
        type="bool",
    ),
        AttributeInfo(
        name="hugepages",
        description="Indicates whether the virtual machines requires hugepages. This is needed for higher performance",
        type="bool",
    ),
        AttributeInfo(
        name="memory",
        description="Specifies the amount of memory needed for the virtual machine to run. The value is in bytes.",
        type="integer",
    ),
        AttributeInfo(
        name="name",
        description="name of the virtual machine instance type",
        type="string",
    ),
        AttributeInfo(
        name="numa",
        description="Indicates whether the virtual machines guest NUMA topology should be mapped the hosts topology. This is needed for higher performance",
        type="bool",
    ),

]


# Description of the loads prefs


prefs_document_content_description = "description of the Virtual Machine Preferences"

prefs_metadata_field_info = [
    AttributeInfo(
        name="clock.preferredTimer.hpet.present",
        description="Indicates whether HPET timer is required",
        type="bool",
    ),
        AttributeInfo(
        name="firmware.preferredUseSecureBoot",
        description="Indicated whether secure boot should be used",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="devices.preferredNetworkInterfaceMultiQueue",
        description="optionally enables the vhost multiqueue feature for virtio interfaces",
        type="bool",
    ),
        AttributeInfo(
        name="devices.preferredInputBus",
        description="optionally defines the preferred bus for Input devices",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="name",
        description="The name of the virtual machine preference",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="kind",
        description="The kind of the virtual machine preference",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="devices.preferredInterfaceModel",
        description="optionally defines the preferred model to be used by Interface devices",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="preferredTerminationGracePeriodSeconds",
        description="Grace period observed after signalling a VirtualMachineInstance to stop after which the VirtualMachineInstance is force terminated",
        type="integer",
    ),
        AttributeInfo(
        name="devices.preferredDiskDedicatedIoThread",
        description="optionally enables dedicated IO threads for Disk devices",
        type="bool",
    ),
        AttributeInfo(
        name="firmware.preferredUseEfi",
        description="Indicated whether the EFI boot should be enabled",
        type="bool",
    ),]

# Load LLM


#llama3-70b-8192
#"llama2-70b-4096"
#"gemma-7b-it"
GROQ_LLM = ChatGroq(
            #model="mixtral-8x7b-32768",
            model="llama3-70b-8192",
            temperature=0,
        )
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


vectordbInstTypes = Chroma(persist_directory=persist_directory, embedding_function=e, collection_name = 'instanceTypes')
vectordbPrefs = Chroma(persist_directory=persist_directory, embedding_function=e, collection_name = 'prefs')


retrieverInstTypes = SelfQueryRetriever.from_llm(
    #GROQ_LLM,
    llm,
    vectordbInstTypes,
    instTypes_document_content_description,
    instTypes_metadata_field_info,
    verbose=True,
    #enable_limit=True,
    #search_kwargs={"k":10},
    #structured_query_translator=QdrantTranslator(metadata_key="metadata"),
    structured_query_translator = ChromaTranslator(),
    fix_invalid=True,

)


retrieverPrefs = SelfQueryRetriever.from_llm(
    llm,
    vectordbPrefs,
    prefs_document_content_description,
    prefs_metadata_field_info,
    verbose=True,
    #enable_limit=True,
    #search_kwargs={"k":10},
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
    rewriting_prompt = PromptTemplate.from_template(
      "Rewrite the following query by extracting the specific requirements for defining a virtual machine from the use query. Use only the words from the use query. Do not add any new requirements: {query}"
    )

    # LLM Chain for query rewriting
    rewriting_chain = rewriting_prompt| GROQ_LLM| StrOutputParser()

    # Run the chain
    definition = await rewriting_chain.ainvoke({"query": definition})
    results = await generate_vm.ainvoke({"request": definition})
    
    return {
        **state,
        "definition": definition,
        "virtualMachine": results,
    }

async def retrieve_instance_type(state: VmCreationState):
    definition = state["definition"]
    vm = state["virtualMachine"]
    try:
        CONTEXT = await retrieverInstTypes.ainvoke(definition)
        print("CONTEXT try1: ", CONTEXT)
    except:
        CONTEXT=[]
    if len(CONTEXT) == 0:
        CONTEXT = vectordbInstTypes.similarity_search(definition, k=1)
        print("CONTEXT try2: ", CONTEXT)
        
    result = await gen_instTypes_chain.ainvoke({"query": definition, "context":CONTEXT})

    print(result.instanceTypes[0].name)

    if result is None:
        result = {"instanceTypes": [{"name": "no instance type"}]}

    vm.spec.instancetype.name = result.instanceTypes[0].name
    return {
        **state,
        "virtualMachine": vm,
    }

  
async def retrieve_preference(state: VmCreationState):
    definition = state["definition"]
    vm = state["virtualMachine"]

    try:
        CONTEXT = await retrieverPrefs.ainvoke(definition)
        print(CONTEXT)
    except:
        CONTEXT=[]
    if len(CONTEXT) == 0:
        CONTEXT = vectordbPrefs.similarity_search(definition, k=1)
        print(CONTEXT)
    
    result = await gen_preferences_chain.ainvoke({"query": definition, "context":CONTEXT})
    if result is None:
        result = {"preferences": [{"name": "no preference"}]}

    vm.spec.preference.name = result.preferences[0].name
    return {
        **state,
        "virtualMachine": vm,
    }


vm_builder = StateGraph(VmCreationState)

nodes = [
    ("init_vm", generate_draft_vm),
    ("retrieve_instance_type", retrieve_instance_type),
    ("retrieve_preference", retrieve_preference),
]
for i in range(len(nodes)):
    name, node = nodes[i]
    vm_builder.add_node(name, node)
    if i > 0:
        vm_builder.add_edge(nodes[i - 1][0], name)

vm_builder.set_entry_point(nodes[0][0])
vm_builder.set_finish_point(nodes[-1][0])
vm = vm_builder.compile(debug=True)

##vm.get_graph().print_ascii()

#results = None

#for step in vm.stream(
#    {
#        "definition": "generate a configuration for a large virtual machine, more than 4 CPUs, meant for high performance, running fedora image.",
#    }
#):
#    name = next(iter(step))
#    print(name)
#    print("-- ", str(step[name])[:300])
#    print("END in step: ", END in step)
    #if END in step:
#    results = step
    #print(step.values())  

#print("Results:", results)
#print("VM: ", results["retrieve_preference"]["virtualMachine"])


