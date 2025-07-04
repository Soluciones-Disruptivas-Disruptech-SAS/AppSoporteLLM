# %% [markdown]
# ### Ejecutar RAG

# %%
import os
import tempfile
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Azure OpenAI Config
AZURE_OPENAI_API_BASE = "https://disruptech-azure-ai.cognitiveservices.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME_CHAT = "gpt-4o-mini"
DEPLOYMENT_NAME_EMBEDDING = "text-embedding-ada-002"

# Pinecone API Key
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")

class PineconeIndexFactory:
    @classmethod
    def get_index(cls, country: str):
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = f"rag-index-{country.lower()}"
        if not pc.has_index(index_name):
            pc.create_index(
                index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"🆕 Índice creado: {index_name}")
        else:
            print(f"📦 Índice existente: {index_name}")
        return pc.Index(index_name), index_name


class RagSystem:
    def __init__(self, country, folder=None) -> None:
        index_instance, index_name = PineconeIndexFactory.get_index(country=country)

        self.files_folder = folder
        self.index_instance = index_instance
        self.index_name = index_name

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_API_BASE,
            openai_api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=DEPLOYMENT_NAME_EMBEDDING
        )

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

        self.config_rag()

    def validate_pdf_format(self, file_input):
        return file_input.content_type == "application/pdf"

    def load_documents(self, documents_load=None) -> List:
        documents = []
        if documents_load is not None:
            for document in documents_load:
                if self.validate_pdf_format(document):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        document.save(tmp_file.name)
                        loader = PyPDFLoader(tmp_file.name)
                        documents.extend(loader.load())
        return documents

    def split_documents(self, documents_to_split=None, chunk_size=500, chunk_overlap=50):
        if documents_to_split is not None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            return text_splitter.split_documents(documents_to_split)

    def create_vectorstore(self) -> None:
        if self.vectorstore is None:
            self.vectorstore = PineconeVectorStore(
                index=self.index_instance,
                embedding=self.embeddings
            )

    def add_documents_to_vectorstore(self, documents_to_add=None):
        if documents_to_add is None:
            return "No se encontraron documentos"

        documents = self.load_documents(documents_to_add)
        split_docs = self.split_documents(documents_to_split=documents)

        if self.vectorstore is None:
            self.create_vectorstore()

        self.vectorstore.add_documents(documents=split_docs)

        return f"{len(split_docs)} documentos agregados exitosamente."

    def create_prompt(self):
        instructions = """
Eres un asistente técnico especializado en desarrollo de software, automatización industrial y sistemas logísticos WCS.

Responde únicamente utilizando la información de los documentos proporcionados a continuación (contexto). No inventes información ni uses conocimientos externos. Si no encuentras la respuesta en el contexto, responde con: “Lo siento, no encontré información relevante en los documentos cargados.”

📄 Documentos disponibles:
{context}

Al final de tu respuesta incluye una sección de referencias con los nombres de los archivos y páginas utilizados en este formato:

📖 Fuentes: [nombre_del_documento.pdf, página X]

Responde de manera técnica, precisa y clara para ingenieros y desarrolladores. Evita creatividad innecesaria y mantente estrictamente en el marco técnico.
"""
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", instructions),
            ("human", "{question}")
        ])
        print("Variables requeridas por el prompt:", chat_prompt.input_variables)
        return chat_prompt

    def create_qa_system(self):
        self.prompt = self.create_prompt()
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_API_BASE,
            openai_api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=DEPLOYMENT_NAME_CHAT,
            temperature=0.5  # 👈 No creatividad, solo facts
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def config_rag(self):
        self.create_vectorstore()
        self.create_qa_system()

    def query_rag(self, question, chat_history=None):
        query = f"""
Historial: {chat_history}
Responde a la siguiente pregunta usando solo los documentos cargados: {question}
"""
        result = self.qa_chain.invoke(query)

        print("📄 CONTENIDO DEL CONTEXTO (documentos recuperados):")
        for i, doc in enumerate(result["source_documents"]):
            meta = doc.metadata
            print(f"\n--- Documento #{i+1} ---")
            print(f"📄 Fuente: {meta.get('source')} | Página: {meta.get('page', '?')}")
            print(doc.page_content[:1000])

        return result["result"]


# %%
from flask import Flask, request, jsonify
from uuid import uuid4

app = Flask(__name__)

# ============================
# Inicializar RAG una sola vez
# ============================
rag_system = RagSystem("soporte")  # Index "rag-index-soporte"
conversaciones = {}  # Historial por sesión (opcional para multiuser)

@app.route('/rag', methods=['POST'])
def rag():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        chat_history = data.get("chat_history", "").strip()

        if not question:
            return jsonify({"error": "No se recibió ninguna pregunta."}), 400

        print(f"💬 Pregunta recibida: {question}")

        # Llamar al RAG
        answer = rag_system.query_rag(question, chat_history)

        response = {
            "answer": answer,
            "session_id": data.get("session_id", str(uuid4()))  # para front multi-session
        }

        print("✅ Respuesta enviada al front")
        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error en /rag: {e}")
        return jsonify({"error": "Error interno en el servidor."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)



