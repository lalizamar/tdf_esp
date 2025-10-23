# -*- coding: utf-8 -*-
# TF-IDF en Español — Estaciones & Clima • UI creativa + fix session_state
# Requisitos: streamlit, scikit-learn, pandas, nltk

import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer

# ─────────────────────────────────────────────────────────
# Config y paletas
st.set_page_config(page_title="TF-IDF en Español — Estaciones", page_icon="🌦️", layout="centered")

SEASONS = {
    "Primavera 🌸": {
        "bg1": "#fff7fb", "bg2": "#eafff5",
        "card": "rgba(255,255,255,0.94)", "accent": "#ff7ac8", "accent2": "#b8f2cf",
        "text": "#1a1a1a", "chip": "#ffe6f4",
        "emoji": "🌸", "story": (
            "Los cerezos despiertan y el aire huele a libro recién abierto. "
            "En los parques nacen historias que zumban como abejas."
        )
    },
    "Verano ☀️": {
        "bg1": "#fffbe6", "bg2": "#ffe7b7",
        "card": "rgba(255,255,255,0.94)", "accent": "#ff9f1a", "accent2": "#ffd166",
        "text": "#1a1a1a", "chip": "#fff0c2",
        "emoji": "🌞", "story": (
            "El sol se estira sin prisa. Chapuzones, pelotas, risas largas: "
            "las frases toman color de mango y limonada."
        )
    },
    "Otoño 🍂": {
        "bg1": "#fff3e6", "bg2": "#ffe0c7",
        "card": "rgba(255,255,255,0.94)", "accent": "#c26d3f", "accent2": "#f5c49b",
        "text": "#1a1a1a", "chip": "#ffe9d7",
        "emoji": "🍁", "story": (
            "Las hojas viajan como cartas antiguas. El viento dobla esquinas y "
            "los sustantivos crujen deliciosamente."
        )
    },
    "Invierno ❄️": {
        "bg1": "#eef7ff", "bg2": "#e8eaff",
        "card": "rgba(255,255,255,0.94)", "accent": "#5aa9ff", "accent2": "#cfe3ff",
        "text": "#1a1a1a", "chip": "#e6f1ff",
        "emoji": "❄️", "story": (
            "La ciudad canta en voz baja. Entre bufandas y aliento de nube, "
            "las palabras se abriguen unas a otras."
        )
    },
}

with st.sidebar:
    st.markdown("## 🎨 Estación / tema")
    season = st.selectbox("Elige el tema visual", list(SEASONS.keys()), index=0)
S = SEASONS[season]

# CSS
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, {S['bg1']} 0%, {S['bg2']} 100%) !important;
}}
[data-testid="stSidebarContent"] {{
  background:{S['card']}; border:2px solid {S['accent2']}; border-radius:18px; padding:.9rem;
  box-shadow:0 10px 26px rgba(0,0,0,.06);
}}
h1,h2,h3,label,p,span,div {{ color:{S['text']} !important; }}
.card {{
  background:{S['card']}; border:2px solid {S['accent2']}; border-radius:22px; padding:1rem 1.1rem;
  box-shadow:0 12px 28px rgba(0,0,0,.08); backdrop-filter:blur(6px);
}}
.chip {{
  display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .7rem; border-radius:999px;
  background:{S['chip']}; border:1.5px solid {S['accent2']}; font-weight:700; font-size:.8rem; margin:.15rem .25rem 0 0;
}}
.btn-primary button {{
  background:{S['accent']} !important; color:#fff !important; border:none !important;
  border-radius:16px !important; padding:.6rem 1rem !important; font-weight:800 !important;
  box-shadow:0 8px 16px rgba(0,0,0,.08) !important;
}}
.progress {{ width:100%; height:12px; border-radius:999px; background:#eee; overflow:hidden; border:2px solid {S['accent2']}; }}
.fill {{ height:100%; background:{S['accent']}; transition:width .4s ease; }}
.small-note {{ opacity:.7; font-size:.9rem }}
</style>
""", unsafe_allow_html=True)

# Encabezado + narrativa
st.markdown(f"""
<div class="card">
  <h1 style="margin:0">{S['emoji']} Demo TF-IDF en Español</h1>
  <p class="small-note" style="margin:.3rem 0 0 0"><b>Crónica climática de palabras — {season}</b></p>
  <p style="margin:.2rem 0 0 0">{S['story']}</p>
  <div>
    <span class="chip">☁️ Documentos = observaciones del día</span>
    <span class="chip">🧭 TF-IDF = intensidad léxica</span>
    <span class="chip">🧠 Cosine = dirección de la respuesta</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Documentos base
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer.
El viento sopla frío cerca del río en invierno.
En verano, la gente nada y toma sol junto al lago.
En otoño, las hojas caen y cubren los senderos del parque.
En primavera florecen los cerezos del barrio.
"""

# Stemmer español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Sugerencias por estación
SUGGESTED = {
    "Primavera 🌸": ["¿Qué florece en el barrio?", "¿Quién canta al amanecer?", "¿Dónde juegan en primavera?"],
    "Verano ☀️": ["¿Dónde nada la gente?", "¿Quién juega bajo el sol?", "¿Qué pasa junto al lago?"],
    "Otoño 🍂": ["¿Qué cae y cubre los senderos?", "¿Dónde se juntan las hojas?", "¿Qué hacen en el parque en otoño?"],
    "Invierno ❄️": ["¿Dónde sopla el viento frío?", "¿Qué sucede cerca del río en invierno?", "¿Quiénes juegan cuando hace frío?"],
}

# Estado seguro para pregunta
if "question_es" not in st.session_state:
    st.session_state["question_es"] = "¿Dónde juegan el perro y el gato?"

# Callback para fijar pregunta (FIX: usar on_click)
def set_question(q: str):
    st.session_state["question_es"] = q

# Layout principal
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    text_input = st.text_area("📄 Documentos (uno por línea):", default_docs, height=170)
    question = st.text_input("❓ Escribe tu pregunta:", key="question_es")

with col2:
    st.markdown("### 💡 Preguntas sugeridas")
    for q in SUGGESTED[season]:
        st.button(q, key=f"sugg_{season}_{q}", use_container_width=True, on_click=set_question, args=(q,))

st.markdown("<br>", unsafe_allow_html=True)

# Acción principal
analyze_col, _ = st.columns([1, 3])
with analyze_col:
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    run = st.button("🔍 Analizar", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if run:
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if not documents:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not st.session_state["question_es"].strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=1)
        X = vectorizer.fit_transform(documents)

        # Matriz TF-IDF
        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        # Pregunta → vector y similitud
        question_vec = vectorizer.transform([st.session_state["question_es"]])
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = float(similarities[best_idx])

        # Respuesta
        st.markdown("### 🎯 Respuesta")
        st.write(f"**Tu pregunta:** {st.session_state['question_es']}")
        if best_score > 0.01:
            st.success(f"**Respuesta:** {best_doc}")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")

        # Barra de similitud (colores de estación)
        pct = max(0.0, min(1.0, best_score))
        st.markdown(
            f'<div class="progress"><div class="fill" style="width:{pct*100:.1f}%"></div></div>',
            unsafe_allow_html=True
        )
        st.info(f"📈 Similitud: {best_score:.3f}")

        # Pronóstico léxico (top términos del documento ganador)
        row = df_tfidf.iloc[best_idx].sort_values(ascending=False).head(8)
        st.markdown("### ⛅ Pronóstico léxico de la estación")
        st.caption("Los términos con mayor *intensidad TF-IDF* del documento más relevante.")
        for term, val in row.items():
            ancho = int(val * 100)
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:.6rem;margin:.2rem 0">
                    <div style="min-width:110px"><b>{term}</b></div>
                    <div style="flex:1;height:10px;border-radius:999px;background:#eee;overflow:hidden;border:1.5px solid {S['accent2']}">
                        <div style="width:{ancho}%;height:100%;background:{S['accent']};"></div>
                    </div>
                    <div style="min-width:48px;text-align:right">{val:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Ranking completo
        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        }).sort_values("Similitud", ascending=False)
        st.markdown("### 🧭 Puntajes de similitud (ordenados)")
        st.dataframe(sim_df, use_container_width=True)

        # Micro-interacciones por estación
        if season == "Invierno ❄️" and best_score >= 0.5:
            st.snow()
        if season == "Verano ☀️" and best_score >= 0.7:
            st.balloons()
