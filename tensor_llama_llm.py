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

index.storage_context.persist(persist_dir="./vectorindex")
query_engine = index.as_query_engine( text_qa_template = CHAT_TEXT_QA_PROMPT)

def get_query_response(query):
    response = query_engine.query(query)
    return response

def get_chat_response(chat):
    response = llm.complete(chat, stopping_tokens=["\n"])
    return str(response)

def restore_index_store(dir_path):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=dir_path)

# load index
    index = load_index_from_storage(storage_context)
    return index
    

