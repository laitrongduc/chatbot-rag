import os
import pickle
import json
import traceback
import time
import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate

try:
    import tiktoken
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    raise ImportError("Please install the dependencies first.")

from document_indexer import chunk_document


def indexer(doc_path, chunk_size, chunk_step, index_path):
    file_count, texts, metadata_list, chunk_id_to_index = chunk_document(
        doc_path=doc_path,
        chunk_size=chunk_size,
        chunk_step=chunk_step,
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(
        texts=texts,
        metadatas=metadata_list,
        embedding=embeddings,
    )
    vectorstore.save_local(folder_path=index_path)
    with open(os.path.join(index_path, "chunk_id_to_index.pkl"), "wb") as f:
        pickle.dump(chunk_id_to_index, f)


class Chat:
    vectorstore = None

    def __init__(self, index_path):
        self.index_path = index_path
        self._init()

    def _init(self):
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
        if self.vectorstore is None:
            self._init()

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


# Streamed response emulator
def response_generator(response):
    response = response.replace("$", "\$")  # prevent rendering as LaTeX
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def main():
    st.set_page_config(
        page_title="RAG Chatbot 🤖 - Chat with your documents",
        page_icon="🏂",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    with st.sidebar:

        # Reset the conversation
        if st.button("🔴 Reset conversation"):
            st.session_state["messages"] = []

        st.title("Configuration")
        # OPENAI_API_KEY
        openai_api_key = st.text_input(
            "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
        )
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # DOCUMENTS
        st.title("Upload your documents")
        if "doc_path" not in st.session_state:
            st.session_state.doc_path = "/data/"
        if not os.path.exists(st.session_state.doc_path):
            os.makedirs(st.session_state.doc_path)
        # when the user uploads a file, store it in the session state
        uploaded_files = st.file_uploader(
            "Choose a file...", type=["pdf", "docx", "pptx"], accept_multiple_files=True
        )
        if uploaded_files:
            all_files = []
            for file in uploaded_files:
                file_type = file.name.split(".")[-1]
                all_files.append(f"{file.name}")
                file_path = os.path.join(st.session_state.doc_path, file.name)
                # delete the old file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                # save the file to the data folder
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
            st.success(f"Uploaded {len(all_files)} files: {all_files}")

        # EMBEDDING
        st.title("Embedding")
        chunk_size = st.number_input(
            "Chunk size", value=128, min_value=1, max_value=1000
        )
        chunk_step = st.number_input(
            "Chunk step", value=128, min_value=1, max_value=1000
        )
        if "index_path" not in st.session_state:
            st.session_state.index_path = "index"
        if not os.path.exists(st.session_state.index_pathex_path):
            os.makedirs(st.session_state.index_path)
        if st.button("Embed"):
            indexer(
                st.session_state.doc_path,
                chunk_size,
                chunk_step,
                st.session_state.index_path,
            )
            st.success(f"Embedding completed. Saved to {st.session_state.index_path}")

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
            chat = Chat(st.session_state.index_path)
            response = chat(prompt)
            response = st.write_stream(response_generator(response))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
