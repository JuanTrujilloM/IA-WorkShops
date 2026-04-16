import streamlit as st
import tiktoken
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from groq import Groq
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import time

st.set_page_config(
    page_title="Desmontando los LLMs",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Desmontando los LLMs")
st.markdown("**Deep Learning y Arquitecturas Transformer — Universidad EAFIT**")

# ── Sidebar: API Key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("Obtén tu API Key gratis en [console.groq.com](https://console.groq.com)")

    st.divider()
    st.header("🤖 Modelo Groq")
    model_options = [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
    ]
    selected_model = st.selectbox("Modelo (low-cost)", model_options)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Módulo 1: Tokenizador",
    "📐 Módulo 2: Embeddings",
    "💬 Módulo 3: Groq Inferencia",
    "📊 Módulo 4: Métricas",
])

# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 1 — Tokenizador
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔬 Laboratorio del Tokenizador")
    st.markdown(
        "Observa cómo el texto se convierte en **tokens** (IDs numéricos) "
        "usando el tokenizador `cl100k_base` de OpenAI (empleado por GPT-4 y modelos similares)."
    )

    user_text = st.text_area(
        "Ingresa un texto:",
        value="El rey y la reina fueron al mercado con el hombre y la mujer.",
        height=100,
    )

    if st.button("Tokenizar", key="btn_tokenize"):
        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(user_text)
        token_strings = [enc.decode([t]) for t in token_ids]

        # ── Visualización con colores alternos ────────────────────────────
        st.subheader("Texto tokenizado")
        colors = ["#4e79a7", "#f28e2b"]
        html_parts = []
        for i, tok in enumerate(token_strings):
            color = colors[i % 2]
            safe = tok.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(
                f'<span style="background-color:{color};color:white;'
                f'padding:2px 4px;border-radius:4px;margin:1px;'
                f'font-family:monospace;font-size:14px;">{safe}</span>'
            )
        st.markdown(" ".join(html_parts), unsafe_allow_html=True)

        # ── Tabla token → ID ──────────────────────────────────────────────
        st.subheader("Mapeo Token → ID")
        df_tokens = pd.DataFrame({"Token": token_strings, "Token ID": token_ids})
        st.dataframe(df_tokens, use_container_width=True, height=250)

        # ── Métrica comparativa ───────────────────────────────────────────
        st.subheader("Métrica comparativa")
        col1, col2, col3 = st.columns(3)
        col1.metric("Caracteres", len(user_text))
        col2.metric("Tokens", len(token_ids))
        ratio = round(len(user_text) / max(len(token_ids), 1), 2)
        col3.metric("Chars / Token", ratio)

        st.info(
            "**¿Por qué importa?** Los modelos cobran y limitan contexto por tokens, "
            "no por caracteres. El español tiende a generar más tokens que el inglés "
            "porque el vocabulario del tokenizador está optimizado para inglés."
        )

# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 — Embeddings
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📐 Geometría de las Palabras (Embeddings)")
    st.markdown(
        "Los embeddings mapean palabras a vectores en un espacio continuo. "
        "Usando PCA los reducimos a 2D para visualizarlos. "
        "Verificamos si se cumple: **v(rey) − v(hombre) + v(mujer) ≈ v(reina)**"
    )
    st.markdown(
        "_Modelo local: `paraphrase-multilingual-MiniLM-L12-v2` "
        "(no requiere API externa)_"
    )

    default_words = "rey, reina, hombre, mujer, príncipe, princesa, rey, noble"
    words_input = st.text_input(
        "Palabras (separadas por coma):",
        value=default_words,
    )

    @st.cache_resource(show_spinner="Cargando modelo de embeddings…")
    def load_embedding_model():
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    if st.button("Generar embeddings y graficar", key="btn_embed"):
        words = [w.strip() for w in words_input.split(",") if w.strip()]
        if len(words) < 2:
            st.warning("Ingresa al menos 2 palabras.")
        else:
            with st.spinner("Calculando embeddings…"):
                model_emb = load_embedding_model()
                embeddings = model_emb.encode(words)

            # ── PCA a 2D ─────────────────────────────────────────────────
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(embeddings)

            df_emb = pd.DataFrame({
                "Palabra": words,
                "PC1": coords_2d[:, 0],
                "PC2": coords_2d[:, 1],
            })

            # ── Plotly interactivo ────────────────────────────────────────
            fig = px.scatter(
                df_emb, x="PC1", y="PC2", text="Palabra",
                title="Espacio de Embeddings (PCA 2D)",
                template="plotly_dark",
                width=750, height=500,
            )
            fig.update_traces(
                textposition="top center",
                marker=dict(size=12, color="#f28e2b"),
            )
            fig.update_layout(xaxis_title="Componente Principal 1",
                              yaxis_title="Componente Principal 2")

            # ── Vectores de analogía ──────────────────────────────────────
            word_lower = [w.lower() for w in words]
            analogy_words = ["rey", "hombre", "mujer", "reina"]
            has_all = all(w in word_lower for w in analogy_words)

            if has_all:
                idx = {w: word_lower.index(w) for w in analogy_words}
                v_king   = embeddings[idx["rey"]]
                v_man    = embeddings[idx["hombre"]]
                v_woman  = embeddings[idx["mujer"]]
                v_queen  = embeddings[idx["reina"]]

                v_analogy = v_king - v_man + v_woman
                pca_analogy = pca.transform(v_analogy.reshape(1, -1))[0]

                fig.add_trace(go.Scatter(
                    x=[pca_analogy[0]], y=[pca_analogy[1]],
                    mode="markers+text",
                    name="rey−hombre+mujer",
                    text=["rey−hombre+mujer"],
                    textposition="bottom center",
                    marker=dict(size=14, color="#59a14f", symbol="star"),
                ))

                cos_sim = np.dot(v_analogy, v_queen) / (
                    np.linalg.norm(v_analogy) * np.linalg.norm(v_queen)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.success(
                    f"**Similitud coseno** entre `rey−hombre+mujer` y `reina`: "
                    f"**{cos_sim:.4f}** "
                    f"({'✅ Relación verificada' if cos_sim > 0.7 else '⚠️ Relación parcial'})"
                )
            else:
                st.plotly_chart(fig, use_container_width=True)
                st.info(
                    "Para verificar la analogía rey−hombre+mujer≈reina, "
                    "incluye las palabras: **rey, hombre, mujer, reina**."
                )

            st.subheader("Varianza explicada por PCA")
            col1, col2 = st.columns(2)
            col1.metric("PC1", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
            col2.metric("PC2", f"{pca.explained_variance_ratio_[1]*100:.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 — Groq Inferencia
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("💬 Inferencia y Razonamiento (Groq API)")

    if not groq_api_key:
        st.warning("Ingresa tu Groq API Key en la barra lateral para usar este módulo.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Parámetros de generación")

            temperature = st.slider(
                "🌡️ Temperatura",
                min_value=0.0, max_value=2.0, value=0.7, step=0.05,
            )
            if temperature < 0.3:
                st.caption("🔵 **Determinista** — respuestas consistentes y predecibles.")
            elif temperature > 0.7:
                st.caption("🔴 **Creativo** — mayor aleatoriedad y diversidad.")
            else:
                st.caption("🟡 **Balanceado** — mezcla de coherencia y variedad.")

            top_p = st.slider(
                "🎯 Top-P (Nucleus Sampling)",
                min_value=0.0, max_value=1.0, value=0.9, step=0.05,
            )
            st.caption(
                "Top-P controla cuántos tokens del vocabulario se consideran. "
                "Con `top_p=0.1` solo los tokens más probables; con `1.0` todos."
            )

            max_tokens = st.number_input(
                "Max tokens de respuesta", min_value=50, max_value=2048, value=300
            )

        with col_right:
            st.subheader("Prompts")
            system_prompt = st.text_area(
                "System Prompt (Instruction Tuning)",
                value=(
                    "Eres un asistente experto en inteligencia artificial. "
                    "Responde de forma concisa y técnica."
                ),
                height=120,
            )
            user_prompt = st.text_area(
                "User Prompt",
                value="¿Qué es el mecanismo de Self-Attention en los Transformers?",
                height=120,
            )

        st.markdown("---")
        st.markdown(
            "**System Prompt vs User Prompt:** El *system prompt* define la personalidad "
            "y restricciones del modelo (Instruction Tuning), mientras que el *user prompt* "
            "es la consulta específica del usuario."
        )

        if st.button("🚀 Generar respuesta", key="btn_generate"):
            with st.spinner("Generando respuesta con Groq…"):
                try:
                    client = Groq(api_key=groq_api_key)
                    t_start = time.perf_counter()

                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=int(max_tokens),
                    )

                    t_end = time.perf_counter()
                    elapsed = t_end - t_start

                    answer = response.choices[0].message.content
                    usage  = response.usage

                    st.subheader("Respuesta del modelo")
                    st.markdown(answer)

                    # Guardar métricas en session_state para Módulo 4
                    st.session_state["last_response"] = {
                        "elapsed": elapsed,
                        "prompt_tokens":     usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens":      usage.total_tokens,
                        "model":             selected_model,
                        "temperature":       temperature,
                        "top_p":             top_p,
                    }
                    st.success("Respuesta generada. Revisa las métricas en el Módulo 4.")

                except Exception as e:
                    st.error(f"Error al conectar con Groq: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 — Métricas de desempeño
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 Métricas de Desempeño")
    st.markdown(
        "Groq destaca por su velocidad gracias al hardware **LPU** (Language Processing Unit). "
        "Aquí se muestran las métricas de la última inferencia realizada."
    )

    if "last_response" not in st.session_state:
        st.info("Aún no hay métricas. Genera una respuesta en el **Módulo 3** primero.")
    else:
        m = st.session_state["last_response"]
        elapsed          = m["elapsed"]
        prompt_tokens    = m["prompt_tokens"]
        completion_tokens = m["completion_tokens"]
        total_tokens     = m["total_tokens"]

        throughput       = completion_tokens / elapsed if elapsed > 0 else 0
        time_per_token   = (elapsed / completion_tokens * 1000) if completion_tokens > 0 else 0

        # ── KPIs ─────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⏱️ Tiempo total (s)",      f"{elapsed:.2f}")
        col2.metric("🔢 Tokens / s",             f"{throughput:.1f}")
        col3.metric("⚡ ms / token",             f"{time_per_token:.2f}")
        col4.metric("📦 Total tokens",           total_tokens)

        # ── Tokens entrada vs salida ──────────────────────────────────────
        st.subheader("Tokens de entrada vs salida")
        df_tok = pd.DataFrame({
            "Tipo":   ["Entrada (Prompt)", "Salida (Completion)"],
            "Tokens": [prompt_tokens, completion_tokens],
        })
        fig_bar = px.bar(
            df_tok, x="Tipo", y="Tokens", color="Tipo",
            text="Tokens",
            color_discrete_sequence=["#4e79a7", "#f28e2b"],
            template="plotly_dark",
            title="Distribución de Tokens",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Parámetros usados ─────────────────────────────────────────────
        st.subheader("Parámetros de inferencia utilizados")
        st.table({
            "Parámetro":  ["Modelo", "Temperatura", "Top-P"],
            "Valor":      [m["model"], m["temperature"], m["top_p"]],
        })

        # ── Gauge de throughput ───────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=throughput,
            title={"text": "Throughput (tokens/s)"},
            gauge={
                "axis":  {"range": [0, 1000]},
                "bar":   {"color": "#59a14f"},
                "steps": [
                    {"range": [0,   200], "color": "#e15759"},
                    {"range": [200, 500], "color": "#f28e2b"},
                    {"range": [500, 1000],"color": "#59a14f"},
                ],
            },
        ))
        fig_gauge.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.info(
            "**¿Por qué Groq es tan rápido?** A diferencia de GPUs de propósito general, "
            "el LPU de Groq está diseñado específicamente para inferencia secuencial de LLMs, "
            "eliminando los cuellos de botella de memoria y logrando throughputs >500 tokens/s."
        )
