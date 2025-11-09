import streamlit as st
import tempfile
import numpy as np
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai 

# =========================
# CONFIG DEL MODELO Y DEL LLM
# =========================
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "models/gemini-2.5-flash"
# =========================
# FUNCIONES BACKEND
# =========================

def _clean_spaces(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def load_sections(path: str):
    reader = PdfReader(path)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    text = _clean_spaces(text)

    title_pat = r"(\b\d+(?:\.\d+)*\s+[^\\n:]{3,120}:)"
    text = re.sub(rf"(?<!\n){title_pat}", r"\n\1", text)

    regex = re.compile(rf"(?m)^(?P<title>\\d+(?:\\d+)*\\s+[^\\n:]{3,120}:)\s*", flags=re.UNICODE)
    spans = list(regex.finditer(text))

    sections = []
    if spans:
        for i, m in enumerate(spans):
            start = m.end()
            end = spans[i+1].start() if i+1 < len(spans) else len(text)
            title = m.group("title").strip()
            body = text[start:end].strip()
            if len(body) >= 30:
                body = re.sub(r"[ \t]{2,}", " ", body)
                title = re.sub(r"\bBloque\s*\\d*\\.?\s*", "", title).strip()
                sections.append(f"{title} {body}")
    else:
        sections = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 50]

    return sections

def build_index(chunks, model_name=EMB_MODEL_NAME):
    model = SentenceTransformer(model_name)
    X = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    return model, np.array(chunks, dtype=object), X

def retrieve_best_chunk(question: str):
    """
    Paso R (retrieve) del RAG
    """
    model = st.session_state.model
    CHUNKS = st.session_state.CHUNKS
    EMB = st.session_state.EMB

    q = model.encode([question], normalize_embeddings=True)
    sims = (q @ EMB.T).ravel()
    i = int(np.argmax(sims))
    return CHUNKS[i]

def generate_answer(question: str, context: str, api_key: str):
    """
    Toma la pregunta y el contexto recuperado, y genera una respuesta
    con un LLM.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(LLM_MODEL_NAME)

        prompt = f"""
        Basándote única y exclusivamente en el siguiente contexto, responde la pregunta.
        Si la respuesta no se encuentra en el contexto, di "No encontré información sobre eso en el documento."

        **Contexto:**
        {context}

        **Pregunta:**
        {question}

        **Respuesta:**
        """

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        st.error(f"Error al contactar la API de Gemini: {e}")
        return None

# =========================
# STREAMLIT
# =========================
st.set_page_config(page_title="Preguntas al manual", page_icon="📄", layout="centered")
st.title("📄 Preguntas al manual 📄")

if "indexed" not in st.session_state:
    st.session_state.indexed = False

uploaded = st.file_uploader("Subí un archivo PDF para procesarlo:", type=["pdf"])

if uploaded:
    if not st.session_state.indexed:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        st.write("Procesando PDF y construyendo índice... Puede tardar unos segundos ⏳")

        chunks = load_sections(tmp_path)
        st.write(f"✅ Total de secciones detectadas: {len(chunks)}")

        model, CHUNKS, EMB = build_index(chunks)
        st.session_state.model = model
        st.session_state.CHUNKS = CHUNKS
        st.session_state.EMB = EMB
        st.write("✅ Embeddings generados y modelo cargado.")

        st.session_state.indexed = True
        st.success("PDF procesado correctamente. Ya podés hacer preguntas 👇")

if st.session_state.indexed:
    api_key_help = "Podés obtener una API key gratis en [Google AI Studio](https://aistudio.google.com/app/apikey)"
    api_key = st.text_input("Ingresá tu API Key de Google AI Studio:", type="password", help=api_key_help)

    question = st.text_input("Indicá sobre qué tema queres saber más:")

    if st.button("Preguntar") and question.strip():
        if not api_key:
            st.error("Por favor, ingresá tu API Key de Google para continuar.")
        else:
            try:
                st.write("Buscando el contexto más relevante...")
                context_chunk = retrieve_best_chunk(question.strip())

                st.write("Generando respuesta...")
                answer = generate_answer(question.strip(), context_chunk, api_key)

                if answer:
                    st.write("### 🤖 Respuesta:")
                    st.write(answer)

                    with st.expander("Ver la fuente (el chunk recuperado):"):
                        st.text_area("", context_chunk, height=200, disabled=True) # Corregí la advertencia de accesibilidad

            except Exception as e:
                st.error(f"Ocurrió un error al procesar la pregunta: {e}")


with st.expander("ℹ️ Ayuda"):
    st.write("""
    **Cómo funciona (RAG):**
    1. Subís un PDF
    2. El sistema divide el texto en secciones y genera embeddings (índice semántico)
    3. Escribís una pregunta
    4. **(Retrieval):** La IA encuentra la sección (chunk) más parecida del documento.
    5. **(Generation):** La IA le pasa ese chunk (como contexto) a un LLM (Gemini) y le pide que genere una respuesta a tu pregunta basándose en él.

    **Modelos:**
    * **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
    * **Generación:** `gemini-1.5-flash` 
    """) # Corregí la discrepancia del modelo
