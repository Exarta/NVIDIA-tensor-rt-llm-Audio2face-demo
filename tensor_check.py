from llama_index.core import SimpleDirectoryReader
from llama_index.llms.nvidia_tensorrt import LocalTensorRTLLM
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core import PromptTemplate

streaming = True
similarity_top_k = 4
is_chat_engine = False
embedded_model = "WhereIsAI/UAE-Large-V1"
embedded_dimension = 1024



def completion_to_prompt(completion: str) -> str:
    """
    Given a completion, return the prompt using llama2 format.
    """
    return f"<s> [INST] {completion} [/INST] "
llm = LocalTensorRTLLM(
    model_path="./model",
    engine_name="llama_float16_tp1_rank0.engine",
    tokenizer_dir="./model/tokenizer/",
    completion_to_prompt=completion_to_prompt,
)

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbeddings(model_name=embedded_model)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=200)
Settings.num_output = 64
Settings.context_window = 3900

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

documents = SimpleDirectoryReader("./rag").load_data()
index = VectorStoreIndex.from_documents(documents)



# text qa prompt
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...'The context information ...' or anything along those lines.'\n"
        "3. Never list the items rather put them in one sentence\n"
        "4. Do not use any special characters such as newline\n"
        "5. Always reply in a single line string\n"
        "6. The answer should not be in the format of a list of items\n"

    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "You are a sales assistant chatbot that just answers questions from the documents provided\n"
            "---------------------\n"
            "{context_str}"
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]
CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

custom_prompt = PromptTemplate(
    """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
just give a simple one line short reply to the Human\
do not use emojis or emoticons
 
<Chat History>
{chat_history}
 
<Follow Up Message>
{question}
 
<Short Reply>
"""
)
 
# list of `ChatMessage` objects

custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="Give me a one line and concise friendly answer",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Okay, sounds good."),
]
index.storage_context.persist(persist_dir="./vectorindex")
query_engine = index.as_query_engine( text_qa_template = CHAT_TEXT_QA_PROMPT)

# chat_engine = index.as_chat_engine(
#     chat_mode="simple",
#     # memory=memory,
#     system_prompt=(
#         "You are a chatbot, able to have normal interactions, as well as talk, but only in short single sentences"
#     ),
# )


chat_engine = SimpleChatEngine.from_defaults(
    llm=llm,
    # chat_history = custom_chat_history
    )
# chat_engine_rag = index.as_chat_engine(
#     chat_mode="condense_plus_context",
#     memory=memory,
#     llm=llm,
#     context_prompt=(
#         "You are a chatbot, able to have normal interactions, as well as talk"
#         "as a sales assistant for mantis , a VR headsets store."
#         "Here are the relevant documents for the context:\n"
#         "{documents}"
#         "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
#     ),
#     verbose=False,
# )





def get_query_response(query):
    response = query_engine.query(query)

    return response

# def get_chat_response(chat):
#     response = chat_engine.chat(chat)

#     return response

def get_chat_response(chat):
    response = llm.complete(chat, stopping_tokens=["\n"])
    print(type(response))
 
    return str(response)

# def get_rag_chat_response(chat):
#     response = chat_engine_rag.chat(chat)

#     return response

def restore_index_store(dir_path):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=dir_path)

# load index
    index = load_index_from_storage(storage_context)
    return index
    


# messages = [
#     ChatMessage(
#         role="system",
#         content="You are a nice friendly chatbot that does casual friendly conversation and gives single line replies.",
        
#     ),
#     ChatMessage(role="user", content="hello how are you doing?")
# ]
 
# response = llm.chat(messages)
# # resp = llm.complete("hello")
# print(str(response))

# prompt = """You are a nice friendly chatbot that gives single line replies.
 
# User: Hello
# Sytem: Hi. How are you doing today?
 
# User: I am fine. how is the weather?
# System: The weather is really nice today.
 
# User:Should i take an umberella"""
 
# response = llm.complete(prompt, stopping_tokens=["\n"])
# print(response)