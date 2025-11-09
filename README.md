# Bot Q&A de Documentos PDF con RAG y Gemini

Este proyecto es una aplicación web interactiva construida con Streamlit que te permite chatear con tus documentos PDF.

Utiliza una arquitectura **RAG (Retrieval-Augmented Generation)**. 
Cuando un usuario sube un documento, la aplicación lo procesa, lo divide en secciones y genera un índice semántico. 
Luego, utiliza el modelo **Gemini 2.5 Flash** de Google para responder preguntas basándose *única y exclusivamente* en la información contenida en el documento.

## Características Principales

* **Carga de PDF:** Permite al usuario subir cualquier documento en formato PDF.
* **Procesamiento de Texto:** Extrae y limpia el texto del documento usando expresiones regulares.
* **Segmentación (Chunking):** Divide el texto en secciones lógicas (chunks) para su indexación.
* **Búsqueda Semántica:** Utiliza un modelo de `SentenceTransformer` para convertir los chunks y la pregunta del usuario en vectores (embeddings).
* **Generación de Respuesta (RAG):** Encuentra el chunk más relevante para la pregunta y se lo entrega como contexto al LLM (Gemini) para que genere una respuesta.
* **Transparencia:** Muestra al usuario el "chunk" exacto de texto que el modelo utilizó para formular su respuesta, permitiendo una fácil verificación de la fuente.

## Stack Tecnológico

* **Framework Web:** `Streamlit` https://bot-textmining-gzzebeu92lbenzu22hctdj.streamlit.app/
* **Modelo de Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Modelo de Generación (LLM):** `Google Gemini 2.5 Flash` (vía API de Google AI Studio)
* **Manejo de PDF:** `PyPDF2`
* **Manejo de Vectores:** `Numpy`

---

## Cómo ejecutarlo localmente

1.  **Clona el repositorio:**
    ```bash
    git clone [URL-DE-TU-REPO-AQUÍ]
    cd [NOMBRE-DEL-REPO]
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    El archivo `requirements.txt` es crucial.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta la aplicación:**
    ```bash
    streamlit run app.py
    ```

5.  Abre tu navegador y ve a `http://localhost:8501`.



## 💡 Desafíos y Lecciones Aprendidas

* **Preprocesamiento del PDF:** El mayor desafío fue la extracción y limpieza del texto. El documento base contenía tablas, formatos complejos y saltos de línea inconsistentes que dificultaban la correcta segmentación (chunking).
* **Sensibilidad a la Pregunta:** Se observó que pequeños cambios en la formulación de una pregunta (ej. "¿qué es...?" vs. "¿qué significa...?") producían respuestas diferentes. Esto es una limitación del sistema RAG "simple" (Naive RAG) que solo recupera el chunk `Top-K=1`.

## 🔮 Próximos Pasos (Mejoras)

Para evolucionar esta PoC a una herramienta más robusta, se planea:

1.  **Mejorar el Retrieval (Top-K):** Implementar la recuperación de los `K=3` o `K=5` chunks más relevantes para dar un contexto más rico y completo al LLM.
2.  **Abstraer la API Key:** Utilizar `st.secrets` de Streamlit para almacenar la API Key en el backend, eliminando la necesidad de que el usuario la ingrese.
3.  **Chunking Adaptativo:** Reemplazar la estrategia de chunking por regex con un método más robusto, como `RecursiveCharacterTextSplitter`, para manejar PDFs con formatos diversos de manera más eficaz.
