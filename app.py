"""Streamlit chat UI: Hahn Agency Brain (RAG over ChromaDB collection 'hahn').

Premium, responsive chat experience matched to Hahn Agency's brand. Animation /
effect patterns (entrance fades, 3-dot typing pulse, accent-glow focus,
micro-interaction hovers) adapted from the ui-ux-pro-max skill's AI-Native UI,
Motion-Driven, and Modern Dark style guidance, using Hahn's exact palette.
"""

import base64
import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)  # let .env win over any stale OPENAI_API_KEY in the environment

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
N_RESULTS = 8

SUGGESTED = [
    "What services does Hahn offer?",
    "How has Hahn helped energy companies?",
    "What makes Hahn different from other agencies?",
]

SYSTEM_PROMPT = (
    "You are the Hahn Agency Brain, a helpful assistant answering questions about "
    "Hahn Agency from the provided website context and the ongoing conversation. "
    "Follow these rules:\n"
    "1. Always answer every part of the question that the context supports. If a "
    "question has several parts, answer each part you can, clearly and directly, "
    "FIRST, before mentioning anything you cannot cover. Never skip an answerable "
    "part just because another part is missing.\n"
    "2. For any part the context does not cover, never give a blunt refusal. After "
    "giving what you do know, warmly note that you don't have that particular detail "
    "yet and invite them to reach out to the Hahn team. For example: \"The CEO of "
    "Hahn is Jeff Hahn. I don't have details on the company's valuation just yet, but "
    "the Hahn team would be happy to help! Feel free to email them at ask@hahn.agency "
    "or give them a call at +1 (512) 344-2010.\" Vary the wording so it flows "
    "naturally, and never use em dashes in your response.\n"
    "3. Only when the context has nothing relevant to ANY part of the question, "
    "respond with just that warm, friendly message pointing them to ask@hahn.agency "
    "or +1 (512) 344-2010.\n"
    "4. Never invent facts (names, numbers, clients, dates, valuations) that are not "
    "in the context.\n"
    "Be clear, helpful, warm, and concise."
)

# --- Theme palettes -----------------------------------------------------------
# Accent maroon stays constant across modes; surfaces/text invert.
THEMES = {
    "dark": {
        "bg": "#0f242c",
        "surface": "#17323c",
        "user_bg": "#14303a",
        "asst_bg": "#1b3a45",
        "input_bg": "#16333d",
        "accent": "#f43737",
        "accent_hover": "#ff5252",
        "text": "#e9eef1",
        "muted": "rgba(233, 238, 241, 0.58)",
        "muted2": "rgba(233, 238, 241, 0.42)",
        "hairline": "rgba(233, 238, 241, 0.10)",
        "header_grad": "linear-gradient(135deg, #173640 0%, #1f4451 48%, #102832 100%)",
        "scroll_thumb": "#2b4a56",
        "pill_bg": "rgba(233, 238, 241, 0.05)",
        "asst_accent": "#00edff",  # cyan accent on bot replies in dark mode (matches bot icon)
        "bot_icon": "#00edff",
        "bot_icon_fg": "#0f242c",
    },
    "light": {
        "bg": "#f4efe7",
        "surface": "#ffffff",
        "user_bg": "#fbf3ee",
        "asst_bg": "#ffffff",
        "input_bg": "#ffffff",
        "accent": "#f43737",
        "accent_hover": "#d92b2b",
        "text": "#201919",
        "muted": "rgba(32, 25, 25, 0.60)",
        "muted2": "rgba(32, 25, 25, 0.45)",
        "hairline": "rgba(32, 25, 25, 0.12)",
        "header_grad": "linear-gradient(135deg, #fde8e8 0%, #fbdcdc 48%, #fdeeee 100%)",
        "scroll_thumb": "#e7c9c9",
        "pill_bg": "rgba(244, 55, 55, 0.05)",
        "asst_accent": "#0f242c",  # blue accent on bot replies in light mode
        "bot_icon": "#0f242c",
        "bot_icon_fg": "#ffffff",
    },
}


@st.cache_data
def logo_data_uri(theme):
    """Return a base64 data URI for the theme's Hahn logo, or None if missing."""
    fname = "hahn-logo-white.png" if theme == "dark" else "hahn-logo-black.png"
    path = Path(__file__).parent / "assets" / fname
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


@st.cache_resource
def get_clients():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_collection("hahn")
    return openai_client, collection


openai_client, collection = get_clients()

st.set_page_config(
    page_title="Hahn Agency Brain",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit's default chrome up front. Use display:none (not visibility:hidden)
# so the header reserves no space, otherwise it leaves a large gap at the top.
st.markdown(
    """
    <style>
    #MainMenu {display: none;}
    footer {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    header, [data-testid="stHeader"] {display: none;}
    /* Collapse the empty wrappers around injected <style> blocks so they
       don't add vertical gaps at the top of the page. */
    [data-testid="stElementContainer"]:has(style) {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "theme" not in st.session_state:
    st.session_state.theme = "light"

T = THEMES[st.session_state.theme]


# --- Injected CSS: theme, responsiveness, animations, micro-interactions ------
def inject_css(t):
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {{
            --bg: {t['bg']};
            --surface: {t['surface']};
            --user-bg: {t['user_bg']};
            --asst-bg: {t['asst_bg']};
            --input-bg: {t['input_bg']};
            --accent: {t['accent']};
            --accent-hover: {t['accent_hover']};
            --text: {t['text']};
            --muted: {t['muted']};
            --muted2: {t['muted2']};
            --hairline: {t['hairline']};
            --pill-bg: {t['pill_bg']};
            --asst-accent: {t['asst_accent']};
            --bot-icon: {t['bot_icon']};
            --bot-icon-fg: {t['bot_icon_fg']};
            --easing: cubic-bezier(0.16, 1, 0.3, 1);
            --title-size: 1.95rem;            /* header title font (fallback text) */
            --logo-height: 2.6rem;            /* Hahn logo height; toggle tracks this */
            --header-pad: clamp(18px, 3vw, 28px);
        }}

        @keyframes hahnFadeUp {{
            from {{ opacity: 0; transform: translateY(14px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes hahnFadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        @keyframes hahnTyping {{
            0%, 60%, 100% {{ transform: translateY(0); opacity: 0.35; }}
            30%           {{ transform: translateY(-5px); opacity: 1; }}
        }}

        /* Fluid root font-size: everything sized in rem scales with the viewport.
           ~15px on phones up to ~20px on large monitors. */
        html {{ font-size: clamp(15px, 0.5vw + 12px, 20px); }}

        .stApp {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 1rem;
        }}

        /* Hide Streamlit chrome: top toolbar (Deploy + 3-dots), sidebar, footer. */
        header[data-testid="stHeader"] {{ display: none !important; }}
        [data-testid="stToolbar"] {{ display: none !important; }}
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
        #MainMenu, footer {{ display: none !important; }}
        /* Remove the "Press Enter to submit" hint (submit button covers it). */
        [data-testid="InputInstructions"] {{ display: none !important; }}

        /* Responsive main column: centered, fluid, room for the fixed bottom bar. */
        .block-container, [data-testid="stMainBlockContainer"] {{
            max-width: min(1700px, 90vw);
            padding-top: 1rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
            padding-bottom: 220px;
        }}

        /* ---------- Theme toggle: anchored to the header box top-right ---------- */
        .st-key-headerbox {{ position: relative; }}
        /* Container spans the title's line height and centers the button on it,
           so the toggle lines up vertically with the "Hahn Agency Brain" text. */
        .st-key-headerbox .st-key-themetoggle {{
            position: absolute; right: clamp(16px, 3vw, 24px);
            /* +1.2rem compensates for the wrapper offset above the card content,
               so the toggle centers on the logo. */
            top: calc(var(--header-pad) + 1.2rem);
            height: var(--logo-height);
            display: flex; align-items: center;
            z-index: 6; width: auto; margin: 0;
        }}
        .st-key-themetoggle button {{
            background: rgba(0, 0, 0, 0.18);
            border: 1px solid var(--hairline);
            border-radius: 50%;
            width: calc(var(--logo-height) * 0.8); height: calc(var(--logo-height) * 0.8);
            min-width: 1.8rem; min-height: 1.8rem;
            padding: 0; line-height: 1;
            font-size: calc(var(--logo-height) * 0.42);
            color: var(--text);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            transition: all 0.2s var(--easing);
        }}
        .st-key-themetoggle button:hover {{
            border-color: var(--accent);
            transform: translateY(-1px) rotate(-12deg);
            box-shadow: 0 0 18px rgba(244, 55, 55, 0.4);
        }}

        /* ---------- Styled header with subtle maroon gradient ---------- */
        .hahn-header {{
            background: {t['header_grad']};
            border: 1px solid var(--hairline);
            border-left: 4px solid var(--accent);
            border-radius: 16px;
            padding: var(--header-pad) clamp(20px, 4vw, 32px);
            margin: 0 0 22px 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20), 0 0 40px rgba(244, 55, 55, 0.10);
            animation: hahnFadeUp 0.6s var(--easing) both;
        }}
        .hahn-header h1 {{
            margin: 0; font-weight: 700; letter-spacing: -0.02em; color: var(--text);
            font-size: var(--title-size); line-height: 1.2;
            display: flex; align-items: center; gap: 0.55rem;
            padding-right: 3rem;  /* keep the logo clear of the toggle */
        }}
        /* Hahn logo image (replaces the brain + title). */
        .hahn-header .hahn-logo-img {{
            height: var(--logo-height); width: auto; max-width: 100%; display: block;
        }}
        .hahn-header .hahn-logo {{ flex: 0 0 auto; line-height: 1; }}
        .hahn-header p {{
            margin: 9px 0 0 0; font-weight: 400; color: var(--muted);
            font-size: 0.92rem; letter-spacing: 0.01em;
        }}

        /* ---------- Chat messages: fade-in + branded cards ---------- */
        [data-testid="stChatMessage"] {{
            background: var(--surface);
            border: 1px solid var(--hairline);
            border-radius: 14px;
            padding: 8px 20px;
            margin-bottom: 16px;
            animation: hahnFadeUp 0.45s var(--easing) both;
        }}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
            background: var(--user-bg);
            border-left: 4px solid var(--accent);
        }}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
            background: var(--asst-bg);
            border-left: 4px solid var(--asst-accent);
        }}
        /* Bot avatar: blue in light mode, cyan in dark mode (override Streamlit's orange). */
        [data-testid="stChatMessageAvatarAssistant"] {{
            background: var(--bot-icon) !important;
            color: var(--bot-icon-fg) !important;
        }}
        /* Force message text to follow the active theme (beats baked-in config textColor). */
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {{
            color: var(--text) !important;
        }}
        /* Comfortable, readable message text that scales with the viewport. */
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
            font-size: 1.1rem; line-height: 1.72;
        }}

        /* ---------- Animated typing indicator (3 pulsing dots) ---------- */
        /* Height matches the 2rem assistant avatar so the dots center on it. */
        .hahn-typing {{ display: flex; align-items: center; gap: 7px; height: 2rem; padding: 0 2px; }}
        .hahn-typing span {{
            width: 9px; height: 9px; border-radius: 50%;
            background: var(--accent-hover); display: inline-block;
            animation: hahnTyping 1.3s infinite var(--easing);
        }}
        .hahn-typing span:nth-child(2) {{ animation-delay: 0.18s; }}
        .hahn-typing span:nth-child(3) {{ animation-delay: 0.36s; }}

        /* ---------- Sources expander: themed, no white flash ---------- */
        [data-testid="stExpander"] {{
            background: var(--surface);
            border: 1px solid var(--hairline) !important;
            border-radius: 12px; overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{ color: var(--muted) !important; font-size: 0.82rem; }}
        [data-testid="stExpander"] summary:hover {{ color: var(--accent-hover) !important; }}
        [data-testid="stExpander"] details > div {{ background: var(--surface) !important; }}
        /* Wrap long source URLs so they never get cut off (esp. on phones). */
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] a {{
            overflow-wrap: anywhere; word-break: break-word;
        }}

        /* ---------- Fixed bottom bar: suggestion pills + trash + input ---------- */
        .st-key-inputbar {{
            position: fixed; left: 0; right: 0; bottom: 12px; margin: 0 auto;
            width: min(1700px, 90vw); z-index: 999;
            background: var(--bg);
            padding-top: 10px;
            animation: hahnFadeIn 0.5s var(--easing) both;
        }}
        /* Soft fade so messages scroll out cleanly behind the bar. */
        .st-key-inputbar::before {{
            content: ""; position: absolute; left: 0; right: 0; top: -28px; height: 28px;
            background: linear-gradient(to bottom, rgba(0,0,0,0), var(--bg));
            pointer-events: none;
        }}
        .st-key-inputbar [data-testid="stHorizontalBlock"] {{ align-items: center; }}
        /* Keep trash + input + send on one line even on mobile (pills still stack). */
        .st-key-inputrow [data-testid="stHorizontalBlock"] {{
            flex-direction: row !important; flex-wrap: nowrap !important; gap: 8px;
        }}
        .st-key-inputrow [data-testid="stColumn"] {{ min-width: 0 !important; }}

        /* Suggestion pills (Try asking). */
        [class*="st-key-sug_"] button {{
            background: var(--pill-bg);
            color: var(--muted);
            border: 1px solid var(--hairline);
            border-radius: 999px;
            font-size: 0.8rem; font-weight: 500;
            padding: 6px 14px; white-space: normal; line-height: 1.25;
            transition: all 0.18s var(--easing);
        }}
        [class*="st-key-sug_"] button:hover {{
            background: var(--accent); color: #fff;
            border-color: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(244, 55, 55, 0.32);
        }}
        [class*="st-key-sug_"] button:active {{ transform: scale(0.97); }}

        /* ---------- Welcome state: centered pills when chat is empty ---------- */
        .st-key-welcome {{
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            min-height: calc(100vh - 300px);
            max-width: 760px; margin: 0 auto;
            gap: 1.25rem;
            animation: hahnFadeUp 0.6s var(--easing) both;
        }}
        .st-key-welcome [data-testid="stHorizontalBlock"] {{ width: 100%; }}
        .hahn-welcome-title {{
            font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em;
            color: var(--text); text-align: center; line-height: 1.25;
        }}
        .hahn-welcome-sub {{
            font-size: 0.95rem; color: var(--muted); text-align: center;
            margin-top: 0.45rem;
        }}
        /* Welcome pills sit a touch larger and softer than the inline version. */
        .st-key-welcome [class*="st-key-sug_"] button {{
            font-size: 0.92rem; padding: 12px 18px; border-radius: 14px;
        }}

        /* Trash (clear chat) icon button to the left of the ask bar. */
        .st-key-clear button {{
            background: var(--surface);
            color: var(--muted);
            border: 1px solid var(--hairline);
            border-radius: 12px;
            height: 46px; padding: 0; font-size: 1.05rem;
            transition: all 0.18s var(--easing);
        }}
        .st-key-clear button:hover {{
            background: var(--accent); color: #fff; border-color: var(--accent-hover);
            transform: translateY(-1px); box-shadow: 0 6px 16px rgba(244, 55, 55, 0.3);
        }}
        .st-key-clear button:active {{ transform: scale(0.95); }}

        /* Text input: themed + glowing maroon border on focus. */
        .st-key-inputbar [data-testid="stForm"] {{
            border: none; padding: 0; background: transparent;
        }}
        .st-key-inputbar [data-baseweb="input"],
        .st-key-inputbar [data-baseweb="base-input"] {{
            background: var(--input-bg) !important;
            border-radius: 14px;
            border: 1px solid var(--hairline);
            transition: box-shadow 0.25s var(--easing), border-color 0.25s var(--easing);
        }}
        .st-key-inputbar [data-baseweb="input"]:focus-within {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(244, 55, 55, 0.45), 0 0 22px rgba(244, 55, 55, 0.4);
        }}
        .st-key-inputbar input {{
            background: transparent !important; color: var(--text) !important;
            height: 46px; font-size: 1.05rem;
        }}
        /* Readable placeholder in both themes (was too dark on the blue input). */
        .st-key-inputbar input::placeholder {{ color: var(--muted) !important; opacity: 1; }}
        /* Send button (form submit). */
        .st-key-inputbar [data-testid="stFormSubmitButton"] button {{
            background: var(--accent); color: #fff; border: none;
            border-radius: 12px; height: 46px; padding: 0; font-size: 1.1rem;
            transition: all 0.18s var(--easing);
        }}
        .st-key-inputbar [data-testid="stFormSubmitButton"] button:hover {{
            background: var(--accent-hover);
            box-shadow: 0 6px 16px rgba(244, 55, 55, 0.4);
            transform: translateY(-1px);
        }}
        .st-key-inputbar [data-testid="stFormSubmitButton"] button:active {{ transform: scale(0.95); }}

        /* Scrollbar polish. */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg); }}
        ::-webkit-scrollbar-thumb {{ background: {t['scroll_thumb']}; border-radius: 6px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

        /* Phone tuning: tidy header, square trash + send buttons, shorter input. */
        @media (max-width: 600px) {{
            :root {{ --title-size: 1.35rem; --logo-height: 2rem; }}
            .hahn-header h1 {{ padding-right: 2.4rem; }}

            /* Pin the trash + send columns to a fixed square footprint; the text
               field takes the remaining (shorter) width. The send selector is
               structural (last column inside the form) so it doesn't also match
               the parent input column. */
            [data-testid="stColumn"]:has(.st-key-clear),
            .st-key-inputrow [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
                flex: 0 0 54px !important; width: 54px !important; min-width: 54px !important;
            }}
            .st-key-clear button,
            .st-key-inputbar [data-testid="stFormSubmitButton"] button {{
                width: 54px; height: 54px; min-width: 54px; padding: 0;
                font-size: 1.4rem; border-radius: 12px;
            }}
            /* Match the input box height to the square buttons. */
            .st-key-inputbar [data-baseweb="input"],
            .st-key-inputbar [data-baseweb="base-input"] {{
                height: 54px !important; min-width: 0 !important;
            }}
            .st-key-inputbar input {{ height: 54px; }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{ animation: none !important; transition: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(T)

# Styled header with the theme toggle anchored in its top-right corner.
with st.container(key="headerbox"):
    logo_uri = logo_data_uri(st.session_state.theme)
    logo_html = (
        f'<img class="hahn-logo-img" src="{logo_uri}" alt="Hahn" />'
        if logo_uri
        else '<span class="hahn-logo-fallback">Hahn Agency Brain</span>'
    )
    st.markdown(
        f"""
        <div class="hahn-header">
            <h1>{logo_html}</h1>
            <p>Powered using RAG · Built on real Hahn content</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    toggle_label = (
        ":material/light_mode:" if st.session_state.theme == "dark" else ":material/dark_mode:"
    )
    if st.button(toggle_label, key="themetoggle", help="Toggle light / dark mode"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()


# --- Helpers ------------------------------------------------------------------
TYPING_HTML = '<div class="hahn-typing"><span></span><span></span><span></span></div>'


def retrieve(question):
    q_embed = openai_client.embeddings.create(
        model=EMBED_MODEL, input=[question]
    ).data[0].embedding
    res = collection.query(query_embeddings=[q_embed], n_results=N_RESULTS)
    return res["documents"][0], res["metadatas"][0]


def render_sources(sources):
    with st.expander(":material/attach_file: Sources"):
        for src in sources:
            st.markdown(f"- {src}")


def generate_answer(prompt):
    """The user's message is already in history. Show typing, store answer, rerun."""
    # Retrieve on the actual question. For short, vague follow-ups ("tell me more"),
    # also fold in the previous question so context carries over — but never for
    # clear standalone questions, where that would derail the search.
    docs, metas = retrieve(prompt)
    prior_user = [m["content"] for m in st.session_state.messages[:-1] if m["role"] == "user"]
    if prior_user and len(prompt.split()) <= 6:
        aug_docs, aug_metas = retrieve(f"{prior_user[-1]} {prompt}")
        seen = set(docs)
        for d, m in zip(aug_docs, aug_metas):
            if d not in seen:
                docs.append(d)
                metas.append(m)
                seen.add(d)
    context = "\n\n---\n\n".join(docs)
    sources = list(dict.fromkeys(m["url"] for m in metas))  # unique, ordered

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[:-1]:
        api_messages.append({"role": m["role"], "content": m["content"]})
    api_messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {prompt}",
    })

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)  # 3-dot typing indicator
        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=api_messages, temperature=0.2
        )
        answer = completion.choices[0].message.content

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()  # finalize: re-render full history, hide suggestion pills


# --- Render chat history ------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])


# --- Welcome state: suggestion pills centered in the page when chat is empty --
clicked_q = None
if not st.session_state.messages:
    # Trim the bottom padding (normally reserved for chat) so the welcome
    # block can truly center in the open space.
    st.markdown(
        "<style>.block-container { padding-bottom: 2rem !important; }</style>",
        unsafe_allow_html=True,
    )
    with st.container(key="welcome"):
        st.markdown(
            '<div class="hahn-welcome-title">What would you like to know?</div>'
            '<div class="hahn-welcome-sub">Pick a question to get started, '
            'or type your own below.</div>',
            unsafe_allow_html=True,
        )
        pill_cols = st.columns(len(SUGGESTED))
        for i, q in enumerate(SUGGESTED):
            if pill_cols[i].button(q, key=f"sug_{i}", use_container_width=True):
                clicked_q = q

# --- Fixed bottom bar: trash + input form ------------------------------------
with st.container(key="inputbar"):
    with st.container(key="inputrow"):
        trash_col, input_col = st.columns([1, 13])
        with trash_col:
            if st.button(":material/delete:", key="clear", help="Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with input_col:
            with st.form("chatform", clear_on_submit=True, border=False):
                field_col, send_col = st.columns([13, 1])
                with field_col:
                    typed = st.text_input(
                        "message",
                        label_visibility="collapsed",
                        placeholder="Ask about Hahn Agency...",
                    )
                with send_col:
                    sent = st.form_submit_button(":material/send:", use_container_width=True)

# --- Generate the answer for a just-asked question ----------------------------
# Runs after the input bar is rendered (so it stays visible during generation)
# and only once the user message is already in history, so the welcome state is
# already gone — the question and typing indicator appear with no overlap.
if st.session_state.get("pending_answer"):
    generate_answer(st.session_state.pop("pending_answer"))

# --- Dispatch: capture a new question, show it instantly, then rerun ----------
prompt = clicked_q or (typed.strip() if sent and typed and typed.strip() else None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_answer = prompt
    st.rerun()
