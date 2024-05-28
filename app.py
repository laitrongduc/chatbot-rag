import streamlit as st
import os
import time
import random
import re
import pickle
import json
import traceback

from langchain_openai import ChatOpenAI

try:
    import tiktoken
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    raise ImportError("Please install the dependencies first.")


from ingest import create_vectorstore
from constant import DOC_PATH, CHUNK_SIZE, CHUNK_STEP, INDEX_PATH
from utils.logging import get_logger

logger = get_logger(__name__)


if not os.path.exists(DOC_PATH):
    os.makedirs(DOC_PATH)
if not os.path.exists(INDEX_PATH):
    os.makedirs(INDEX_PATH)


def is_valid_api_key(api_key: str) -> bool:
    """Determine if input is valid OpenAI API key.

    Args:
        api_key (str): An input string to be validated.

    Returns:
        bool: A boolean that indicates if input is valid OpenAI API key.
    """
    api_key_re = re.compile(r"^sk-(proj-)?[A-Za-z0-9]{32,}$")
    return bool(re.fullmatch(api_key_re, api_key))


def response_generator(response):
    response = response.replace("$", "\$")  # prevent rendering as LaTeX
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


class Chat:
    def __init__(self, index_path):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        with open(
            os.path.join(
                self.index_path,
                "chunk_id_to_index.pkl",
            ),
            "rb",
        ) as f:
            self.chunk_id_to_index = pickle.load(f)

        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def __call__(self, query: str, size: int = 5, target_length: int = 512):
        rag_prompt = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use five sentences maximum and keep the answer as concise as possible.

        {context}

        Question: {query}

        Helpful Answer:
        """
        result = self.vectorstore.similarity_search(
            query=query,
            k=size,
        )

        expanded_chunks = self.do_expand(result, target_length)
        context = self.wrap_text_with_delimiter_temporal(
            "\n```json\n" + json.dumps(expanded_chunks, indent=4) + "```\n",
        )

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            # api_key=os.environ["OPENAI_API_KEY"],
        )
        user_message = [rag_prompt.format(context=context, query=query)]

        try:
            response = llm.invoke(user_message)
            return response.content
        except Exception as e:
            traceback.print_exc()
            return f"An error occurred: {e}"

    def wrap_text_with_delimiter_temporal(self, text: str) -> str:
        """wrap text with delimiter"""
        from prompt_util import PromptUtil

        return PromptUtil.wrap_text_with_delimiter(
            text,
            PromptUtil.DELIMITER_TEMPORAL,
        )

    def do_expand(self, result, target_length):
        expanded_chunks = []
        # do expansion
        for r in result:
            source = r.metadata["source"]
            chunk_id = r.metadata["chunk_id"]
            content = r.page_content

            expanded_result = content
            left_chunk_id, right_chunk_id = chunk_id - 1, chunk_id + 1
            left_valid, right_valid = True, True
            chunk_ids = [chunk_id]
            while True:
                current_length = len(self.enc.encode(expanded_result))
                if f"{source}_{left_chunk_id}" in self.chunk_id_to_index:
                    chunk_ids.append(left_chunk_id)
                    left_chunk_index = self.vectorstore.index_to_docstore_id[
                        self.chunk_id_to_index[f"{source}_{left_chunk_id}"]
                    ]
                    left_chunk = self.vectorstore.docstore.search(left_chunk_index)
                    encoded_left_chunk = self.enc.encode(left_chunk.page_content)
                    if len(encoded_left_chunk) + current_length < target_length:
                        expanded_result = left_chunk.page_content + expanded_result
                        left_chunk_id -= 1
                        current_length += len(encoded_left_chunk)
                    else:
                        expanded_result += self.enc.decode(
                            encoded_left_chunk[-(target_length - current_length) :],
                        )
                        current_length = target_length
                        break
                else:
                    left_valid = False

                if f"{source}_{right_chunk_id}" in self.chunk_id_to_index:
                    chunk_ids.append(right_chunk_id)
                    right_chunk_index = self.vectorstore.index_to_docstore_id[
                        self.chunk_id_to_index[f"{source}_{right_chunk_id}"]
                    ]
                    right_chunk = self.vectorstore.docstore.search(right_chunk_index)
                    encoded_right_chunk = self.enc.encode(right_chunk.page_content)
                    if len(encoded_right_chunk) + current_length < target_length:
                        expanded_result += right_chunk.page_content
                        right_chunk_id += 1
                        current_length += len(encoded_right_chunk)
                    else:
                        expanded_result += self.enc.decode(
                            encoded_right_chunk[: target_length - current_length],
                        )
                        current_length = target_length
                        break
                else:
                    right_valid = False

                if not left_valid and not right_valid:
                    break

            expanded_chunks.append(
                {
                    "chunk": expanded_result,
                    "metadata": r.metadata,
                    # "length": current_length,
                    # "chunk_ids": chunk_ids
                },
            )
        return expanded_chunks


def show_ui():
    st.set_page_config(
        page_title="RAG Chatbot 🤖 - Chat with your documents",
        page_icon="🏂",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        # clear chat history
        if st.button("Clear chat history"):
            st.session_state.messages = []
        # configuration
        st.markdown("## Configuration")
        ## OPENAI API KEY
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            if is_valid_api_key(openai_api_key):
                st.success("Valid API key")
                os.environ["OPENAI_API_KEY"] = openai_api_key
            else:
                st.error("Invalid API key")
        ## DOCUMENT
        uploaded_file = st.file_uploader(
            "Upload your document", type=["pdf", "docx", "txt"]
        )
        add_data = st.button("Add data")
        if add_data and uploaded_file:
            with st.spinner("Reading, chunking, and embedding file ...."):
                file_path = os.path.join(DOC_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                logger.info(f"Document uploaded: {uploaded_file.name}")
                vectorstore = create_vectorstore(
                    DOC_PATH, CHUNK_SIZE, CHUNK_STEP, INDEX_PATH
                )
                st.session_state.vs = vectorstore
                st.success("File uploaded, chunked and embedded successfully.")
                logger.info("File uploaded, chunked and embedded successfully.")

    st.title("RAG Chatbot 🤖 - Chat with your documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # response = st.write_stream(response_generator())
            chat = Chat(INDEX_PATH)
            response = chat(prompt)
            response = st.write_stream(response_generator(response))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    show_ui()


if __name__ == "__main__":
    main()
