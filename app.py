"""
app.py — Streamlit chatbot demo.
Loads the quantized GGUF model via inference.py and supports
multi-turn conversation with visible tool-call output.
Runs on Colab CPU runtime out of the box.

Usage:
  streamlit run app.py
  MODEL_PATH=models/model-q4_k_s.gguf streamlit run app.py
"""

import json
import os
import sys
import streamlit as st

# ── project root on path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import run

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Tool-Calling Assistant",
    page_icon="🤖",
    layout="centered",
)

# ─────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────
st.markdown("""
<style>
/* Chat bubbles */
.user-bubble {
    background: #2563eb;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    margin: 6px 0 6px 20%;
    font-size: 0.95rem;
    word-wrap: break-word;
}
.bot-bubble {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px;
    margin: 6px 20% 6px 0;
    font-size: 0.95rem;
    word-wrap: break-word;
}
.tool-badge {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #334155;
    color: #38bdf8;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-family: monospace;
    margin-bottom: 6px;
}
.tool-block {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: monospace;
    font-size: 0.82rem;
    color: #93c5fd;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
TOOL_ICONS = {
    "weather":  "🌤️",
    "calendar": "📅",
    "convert":  "📐",
    "currency": "💱",
    "sql":      "🗄️",
}

def parse_tool_call(response: str):
    """Return (tool, args) dict if response is a tool call, else None."""
    if "<tool_call>" not in response:
        return None
    try:
        raw = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return json.loads(raw)
    except Exception:
        return None


def render_bot_message(response: str):
    """Render a bot response as a styled bubble."""
    parsed = parse_tool_call(response)
    if parsed:
        tool = parsed.get("tool", "unknown")
        args = parsed.get("args", {})
        icon = TOOL_ICONS.get(tool, "🛠️")
        args_str = json.dumps(args, indent=2)
        st.markdown(
            f"""<div class="bot-bubble">
                <div class="tool-badge">{icon} {tool}</div>
                <div class="tool-block">{args_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-bubble">{response}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Tool-Calling Assistant")
    st.caption("Llama-3.2-1B · Q4_K_S · CPU-only")
    st.divider()

    model_path = os.environ.get("MODEL_PATH", "models/model-q4_k_s.gguf")
    model_exists = os.path.exists(model_path)

    if model_exists:
        size_mb = os.path.getsize(model_path) / (1024 ** 2)
        st.success(f"Model loaded ✅\n`{os.path.basename(model_path)}`  {size_mb:.0f} MB")
    else:
        st.info(
            "🎮 **Demo mode active**\n\n"
            "Responses use keyword matching.\n"
            "Train + quantize to load the real model:\n"
            "`make all`"
        )

    st.divider()
    st.markdown("**Available tools**")
    for tool, icon in TOOL_ICONS.items():
        st.markdown(f"{icon} `{tool}`")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Try an example:**")
    examples = [
        "What's the weather in London?",
        "Convert 100 USD to EUR.",
        "How many feet is 50 meters?",
        "Schedule a team meeting on 2026-05-01.",
        "List employees earning over 70k.",
        "Who are you?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state.pending_input = ex
            st.rerun()


# ─────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────
st.markdown("## 💬 Chat")

# Init state
if "messages" not in st.session_state:
    st.session_state.messages = []        # [{"role": "user"|"assistant", "content": str}]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""

# Render history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        render_bot_message(msg["content"])

# ─────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────
user_input = st.chat_input("Type a message…")

# Accept input from sidebar example buttons OR chat_input
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = ""

if user_input:
    # Show user bubble immediately
    st.markdown(
        f'<div class="user-bubble">{user_input}</div>',
        unsafe_allow_html=True,
    )

    # Build history for inference (all prior turns)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Run model
    with st.spinner("Thinking…"):
        try:
            response = run(user_input, history)
        except Exception as e:
            response = f"⚠️ Error: {e}"

    # Persist both turns
    st.session_state.messages.append({"role": "user",      "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Show bot response
    render_bot_message(response)

    st.rerun()
