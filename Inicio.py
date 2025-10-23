# -*- coding: utf-8 -*-
# TF-IDF en Español — Estaciones & Clima Edition
# Requisitos: streamlit, scikit-learn, pandas, nltk

import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer

# ─────────────────────────────────────────────────────────
# Config y Paletas por estación
st.set_page_config(page_title="TF-IDF en Español — Estaciones", page_icon="🌦️", layout="centered")

SEASONS = {
    "Primavera 🌸": {
        "bg1": "#fff7fb", "bg2": "#eafff5",
        "card": "rgba(255,255,255,0.94)", "accent": "#ff7ac8", "accent2": "#b8f2cf",
        "text": "#111", "chip": "#ffe6f4",
        "story": "La ciudad despierta. Brotan flores, vuelan abejas y los parques vuelven a sonar con risas suaves.",
    },
    "Verano ☀️": {
        "bg1": "#fffbe6", "bg2": "#ffe7b7",
        "card": "rgba(255,255,255,0.94)", "accent": "#ff9f1a", "accent2": "#ffd166",
        "text": "#111", "chip": "#fff0c2",
        "story": "El sol se estira hasta tarde. Hay chapuzones, helados y música en plazas calientes de luz.",
    },
    "Otoño 🍂": {
        "bg1": "#fff3e6", "bg2": "#ffe0c7",
        "card": "rgba(255,255,255,0.94)", "accent": "#c26d3f", "accent2": "#f5c49b",
        "text": "#111", "chip": "#ffe9d7",
        "story": "Las hojas giran como cartas antiguas. El viento cuenta secretos y crujen los caminos del parque.",
    },
    "Invierno ❄️": {
        "bg1": "#eef7ff", "bg2": "#e8eaff",
        "card": "rgba(255,255,255,0.94)", "accent": "#5aa9ff", "accent2": "#cfe3ff",
        "text": "#111", "chip": "#e6f1ff",
        "story": "El aire muerde un poquito. La ciudad canta bajito y la nieve dibuja silencios luminosos.",
    },
}

with st.sidebar:
    st.markdown("## 🎨 Estación")
    season = st.selectbox("Elige el tema", list(SEASONS.keys()), index=0)
S = SEASONS[season]

# CSS que sí aplica a Streamlit
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, {S['bg1']} 0%, {S['bg2']} 100%) !important;
}}
[data-testid="stSidebarContent"] {{
  background:{S['card']}; border:2px solid {S['accent2']}; border-radius:18px; padding:.8rem;
  box-shadow:0 10px 26px rgba(0,0,0,.06);
}}
h1,h2,h3,label,p,span,div {{ color:{S['text']} !important; }}
.card {{
  background:{S['card']}; border:2px solid {S['accent2']}; border-radius:22px; padding:1rem 1.1rem;
  box-shadow:0 12px 28px rgba(0,0,0,.08); backdrop-filter:blur(6px);
}}
.chip {{
  display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .7rem; border-radius:999px;
  background:{S['chip']}; border:1.5px solid {S['accent2']}; font-weight:700; font-size:.8rem; margin-right:.25rem;
}}
div.stButton>button {{
  background:{S['accent']}; color:#fff; border:none; border-radius:16px; padding:.6rem 1rem; font-weight:800;
  box-shadow:0 8px 16px rgba(0,0,0,.08); transition:transform .06s ease, filter .2s ease;
}}
div.stButton>button:hover {{ transform: translateY(-1px); filter: brightness(1.06); }}
.stTextArea textarea, .stTextInput input {{
  border-radius:14px !important; border:2px solid {S['accent2']} !important;
}}
.progress {{ width:100%; height:12px; border-radius:999px; background:#eee; overflow:hidden; border:2px solid {S['accent2']}; }}
.fill {{ height:100%; background:{S['accent']}; transition:width .4s ease; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Narrativa climática
st.markdown(f"""
<div class="card">
  <h1 style="margin:0">🔎 Demo TF-IDF en Español</h1>
  <p style="margin:.3rem 0 0 0"><b>Bitácora meteorolingüística — {season}</b></p>
  <p style="margin:.2rem 0 0 0">{S['story']}</p>
  <div class="chip">☁️ Documentos = observaciones del día</div>
  <div class="chip">🧭 TF-IDF = pistas para encontrar la mejor respuesta</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Documentos de ejemplo (ya con vibe de clima/cotidianidad)
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

# Sugerencias por estación (se conectan con la narrativa)
SUGGESTED = {
    "Primavera 🌸": [
        "¿Qué florece en el barrio?",
        "¿Dónde juegan los niños cuando florecen los cerezos?",
        "¿Quién canta al amanecer?"
    ],
    "Verano ☀️": [
        "¿Dónde nada la gente?",
        "¿Qué actividad hacen junto al lago?",
        "¿Quién juega bajo el sol?"
    ],
    "Otoño 🍂": [
        "¿Qué cae y cubre los senderos?",
        "¿Dónde se juntan las hojas?",
        "¿Qué hacen los niños en el parque durante otoño?"
    ],
    "Invierno ❄️": [
        "¿Dónde sopla frío el viento?",
        "¿Qué pasa cerca del río en invierno?",
        "¿Quiénes juegan cuando hace frío?"
    ],
}

# ─────────────────────────────────────────────────────────
# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📄 Documentos (uno por línea):", default_docs, height=170)

    # estado para la pregunta (para que los botones del lado derecho la rellenen)
    if "question_es" not in st.session_state:
        st.session_state.question_es = "¿Dónde juegan el perro y el gato?"

    question = st.text_input("❓ Escribe tu pregunta:", key="question_es")

with col2:
    st.markdown("### 💡 Preguntas sugeridas")
    for q in SUGGESTED[season]:
        if st.button(q, use_container_width=True):
            st.session_state.question_es = q  # se actualiza el input por key
    st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Acción principal (MISMA LÓGICA DEL PROFESOR)
if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )

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
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = float(similarities[best_idx])

        st.markdown("### 🎯 Respuesta")
        st.write(f"**Tu pregunta:** {question}")
        if best_score > 0.01:
            st.success(f"**Respuesta:** {best_doc}")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")

        # Barra de similitud con color de estación
        pct = max(0.0, min(1.0, best_score))
        st.markdown(f'<div class="progress"><div class="fill" style="width:{pct*100:.1f}%"></div></div>', unsafe_allow_html=True)
        st.info(f"📈 Similitud: {best_score:.3f}")

        # Pequeño guiño estacional
        if season == "Invierno ❄️" and best_score >= 0.5:
            st.snow()
        if season == "Verano ☀️" and best_score >= 0.7:
            st.balloons()
