import re
import unicodedata

import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

# ----------------------------- Configuración -----------------------------
st.set_page_config(page_title="Asesora de maquillaje", page_icon="💄")


def get_secret(nombre, default=None):
    try:
        return st.secrets[nombre]
    except Exception:
        return default


API_KEY = get_secret("GEMINI_API_KEY")
MODELO = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
UMBRAL = 0.34  # qué tan parecida debe ser la pregunta a un tema de la guía

SALUDO = (
    "¡Hola! 💖 Soy tu asesora virtual de maquillaje. "
    "Puedo ayudarte a elegir base, tono, máscara de pestañas, labial, "
    "delineador, polvos, brochas y más. **¿Qué quieres consultar hoy?**"
)

PROMPT_GUIA = """Eres una asesora amable de una guía de compra de maquillaje.
Responde la pregunta del cliente usando ÚNICAMENTE la información de la guía
que se te entrega a continuación. Sé clara, cercana y concisa, en español.
Si la guía no cubre algún detalle de la pregunta, dilo con honestidad.

GUÍA:
{contexto}
"""

PROMPT_GENERAL = """Eres una asesora amable y experta en maquillaje y cuidado de la piel.
La pregunta del cliente NO está en la guía de compra de la tienda, así que responde
con tu conocimiento general, en español, de forma clara y concisa.
No inventes marcas, precios ni datos específicos de productos. Si la consulta
no tiene relación con maquillaje o belleza, indícalo con amabilidad y
redirige la conversación a temas de maquillaje.
"""

# ----------------------------- Lectura de la guía -----------------------------
STOPWORDS = set(
    "de la el los las un una unos unas y o a en con para por que como cual cuales "
    "es son mi mis tu tus me se al del lo le les si no mas muy que quiero necesito "
    "puedo debo hay tengo busco recomiendas recomienda".split()
)


def normalizar(texto):
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", texto)


def raices(texto):
    # raíz cruda (5 letras) para que "pestañas" y "pestaña" coincidan
    return {t[:5] for t in normalizar(texto) if t not in STOPWORDS and len(t) > 2}


@st.cache_data
def cargar_guia(ruta="guia_maquillaje.html"):
    with open(ruta, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    temas = []
    for card in soup.select("article.faq-card"):
        q = card.select_one(".faq-question")
        num = q.select_one("span")
        if num:
            num.decompose()
        pregunta = q.get_text(" ", strip=True)
        respuesta = card.select_one(".faq-answer").get_text(" ", strip=True)
        temas.append({"titulo": pregunta, "texto": f"{pregunta}\n{respuesta}"})

    productos = []
    for p in soup.select(".producto-card"):
        productos.append(p.get_text(" ", strip=True))
    if productos:
        temas.append(
            {
                "titulo": "Productos recomendados",
                "texto": "Productos recomendados (precios orientativos):\n- "
                + "\n- ".join(productos),
            }
        )

    for t in temas:
        t["raices"] = raices(t["texto"])
    return temas


def buscar_en_guia(pregunta, temas, max_temas=2):
    q = raices(pregunta)
    if not q:
        return []
    puntuados = []
    for t in temas:
        score = len(q & t["raices"]) / len(q)
        puntuados.append((score, t))
    puntuados.sort(key=lambda x: x[0], reverse=True)
    return [t for s, t in puntuados[:max_temas] if s >= UMBRAL]


# ----------------------------- Gemini -----------------------------
def responder(historial, pregunta, temas_encontrados):
    client = genai.Client(api_key=API_KEY)

    if temas_encontrados:
        contexto = "\n\n".join(t["texto"] for t in temas_encontrados)
        instruccion = PROMPT_GUIA.format(contexto=contexto)
    else:
        instruccion = PROMPT_GENERAL

    # últimos mensajes para mantener el hilo de la conversación
    contenido = []
    for m in historial[-6:]:
        rol = "user" if m["role"] == "user" else "model"
        contenido.append(types.Content(role=rol, parts=[types.Part(text=m["content"])]))
    contenido.append(types.Content(role="user", parts=[types.Part(text=pregunta)]))

    resp = client.models.generate_content(
        model=MODELO,
        contents=contenido,
        config=types.GenerateContentConfig(system_instruction=instruccion),
    )
    return resp.text


# ----------------------------- Interfaz -----------------------------
st.title("💄 Guía de compra de maquillaje")

if not API_KEY:
    st.error("Falta la clave GEMINI_API_KEY en los *secrets* de la app.")
    st.stop()

temas = cargar_guia()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"role": "assistant", "content": SALUDO, "origen": None}
    ]

for m in st.session_state.mensajes:
    with st.chat_message(m["role"], avatar="💖" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        if m.get("origen"):
            st.caption(m["origen"])

pregunta = st.chat_input("Escribe tu consulta sobre maquillaje...")

if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)

    encontrados = buscar_en_guia(pregunta, temas)
    if encontrados:
        origen = "📖 Respuesta basada en la guía de compra"
    else:
        origen = "🌐 Esta consulta no está en la guía; respondí con Gemini en línea"

    with st.chat_message("assistant", avatar="💖"):
        with st.spinner("Pensando..."):
            try:
                texto = responder(st.session_state.mensajes, pregunta, encontrados)
            except Exception as e:
                texto = "Lo siento, tuve un problema al conectarme. Intenta de nuevo en un momento."
                origen = f"⚠️ Error: {e}"
        st.markdown(texto)
        st.caption(origen)

    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    st.session_state.mensajes.append(
        {"role": "assistant", "content": texto, "origen": origen}
    )
