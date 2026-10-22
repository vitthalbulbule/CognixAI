from dotenv import load_dotenv

load_dotenv()

import streamlit as st


from pathlib import Path
from uuid import uuid4

import streamlit as st

from agentic_rag.config import settings
from agentic_rag.graph.build import build_graph
from agentic_rag.ingestion import ensure_index, ingest


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CognixAI | Knowledge Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small UI enhancements
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.25);
        }

        .main-header {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .sub-header {
            color: var(--text-color);
            opacity: 0.75;
            margin-bottom: 1.2rem;
        }

        div[data-testid="stFileUploader"] {
            border-radius: 12px;
        }

        div.stButton > button {
            border-radius: 9px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LANGGRAPH NODE STATUS LABELS
# --------------------------------------------------

NODE_LABELS = {
    "route": "🧭 Routing your question...",
    "retrieve": "📚 Searching the knowledge base...",
    "grade_documents": "🔍 Checking document relevance...",
    "rewrite_query": "✏️ Improving the search query...",
    "web_search": "🌐 Searching the web...",
    "query_sql": "🗄️ Querying the SQL database...",
    "generate": "💬 Generating the answer...",
    "check_hallucinations": "🛡️ Checking answer evidence...",
}

SOURCE_BANNERS = {
    "vectorstore": "📚 Answered using your indexed documents.",
    "web_search": "🌐 Answered using public web search.",
    "sql": "🗄️ Answered using the SQL database.",
}


# --------------------------------------------------
# GRAPH INITIALIZATION
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading models and knowledge base...")
def get_graph():
    ensure_index()
    return build_graph()


# --------------------------------------------------
# SIDEBAR: KNOWLEDGE BASE
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🧭 CognixAI ")
    st.caption("Your multi-source AI knowledge assistant")

    st.divider()

    st.markdown("### 📚 Knowledge Base")
    st.write(
        "Upload documents to build a searchable knowledge base. "
        "The agent can retrieve relevant passages when answering questions."
    )

    uploaded_files = st.file_uploader(
        "Choose your documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, and Markdown.",
        key="knowledge_uploader",
    )

    if uploaded_files:
        st.markdown("**Selected files**")

        total_bytes = sum(file.size for file in uploaded_files)

        for file in uploaded_files:
            file_size_kb = file.size / 1024
            st.caption(f"📄 {file.name} · {file_size_kb:.1f} KB")

        st.caption(
            f"📦 {len(uploaded_files)} file(s) selected · "
            f"{total_bytes / 1024:.1f} KB total"
        )

    index_clicked = st.button(
        "⚡ Save and index documents",
        use_container_width=True,
        type="primary",
    )

    if index_clicked:
        if not uploaded_files:
            st.warning("Please select at least one document first.")
        else:
            try:
                documents_dir = Path(settings.documents_dir).parent / "user_uploads"
                documents_dir.mkdir(parents=True, exist_ok=True)

                saved_files = []

                with st.spinner("📥 Saving uploaded documents..."):
                    for uploaded_file in uploaded_files:
                        # Prevent path traversal and filename collisions.
                        safe_name = Path(uploaded_file.name).name
                        unique_name = f"{uuid4().hex[:8]}_{safe_name}"
                        destination = documents_dir / unique_name

                        destination.write_bytes(uploaded_file.getvalue())
                        saved_files.append(unique_name)

                with st.spinner(
                    "🧠 Creating document embeddings and updating the index..."
                ):
                    chunk_count = ingest(directory=str(documents_dir))

                # Rebuild the cached graph after the knowledge base changes.
                get_graph.clear()

                st.session_state["last_index_result"] = {
                    "files": saved_files,
                    "chunks": chunk_count,
                }

            except Exception as exc:
                st.error(f"❌ Indexing failed: {exc}")

    # Keep the indexing result visible after the button interaction.
    result = st.session_state.get("last_index_result")

    if result:
        st.success(
            f"✅ Indexed {len(result['files'])} file(s) "
            f"into {result['chunks']} text chunks."
        )

        with st.expander("📂 Recently indexed files"):
            for filename in result["files"]:
                st.write(f"📄 {filename}")

    st.divider()

    st.markdown("### ✨ Capabilities")
    st.markdown(
        """
        - 📚 Document retrieval
        - 🔍 Evidence relevance checking
        - ✏️ Query rewriting
        - 🌐 Web search
        - 🗄️ SQL database routing
        """
    )

    st.divider()

    st.markdown("### 📌 Supported formats")
    st.caption("📕 PDF  ·  📝 TXT  ·  📘 Markdown")

    st.caption(
        "Uploaded files are saved in your project's documents folder. "
        "Indexing rebuilds the collection from files in that folder."
    )


# --------------------------------------------------
# MAIN CHAT INTERFACE
# --------------------------------------------------

st.markdown(
    '<div class="main-header">💬 Knowledge Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-header">'
    "Ask questions about your documents or let the agent select "
    "another available information source."
    "</div>",
    unsafe_allow_html=True,
)

# Initialize the graph after any indexing operation.
graph = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# QUESTION AND ANSWER
# --------------------------------------------------

if question := st.chat_input(
    "💭 Ask a question about your documents..."
):
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        state = {}

        try:
            with st.status(
                "🧠 Working on your question...",
                expanded=True,
            ) as status:

                for update in graph.stream(
                    {"question": question},
                    stream_mode="updates",
                ):
                    for node, values in update.items():
                        label = NODE_LABELS.get(
                            node,
                            f"⚙️ Processing: {node}",
                        )

                        st.write(label)

                        if values:
                            state.update(values)

                status.update(
                    label="✅ Answer ready",
                    state="complete",
                    expanded=False,
                )

            answer = state.get(
                "generation",
                "I couldn't generate an answer. Please try again.",
            )

            st.markdown(answer)

            # Indicate which source was used.
            datasource = state.get("datasource", "")
            if banner := SOURCE_BANNERS.get(datasource):
                st.caption(banner)

            # Show retrieved evidence when available.
            evidence = state.get("documents", [])

            if evidence:
                with st.expander(
                    f"📖 View supporting evidence ({len(evidence)} item(s))"
                ):
                    for index, document in enumerate(evidence, start=1):
                        source = document.metadata.get(
                            "source",
                            "Unknown source",
                        )

                        st.markdown(f"**{index}. 📄 {source}**")
                        st.text(document.page_content[:700])

                        st.divider()

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception:
            st.error(
                "Something went wrong while answering your question. "
                "Check the terminal for the detailed error message, "
                "then try again."
            )
            st.exception(Exception("The question-processing workflow failed."))
