import requests
import streamlit as st
from typing import List, Dict

import config

# ---- AI Provider Helpers ----

def call_openai(messages: List[Dict[str, str]]) -> str:
    api_key = config.OPENAI_API_KEY
    if not api_key:
        return "OpenAI API key not configured"
    response = requests.post(
        config.LLM_API_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": config.LLM_MODEL, "messages": messages, "temperature": 0.3},
        timeout=15,
    )
    if response.status_code == 200:
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return f"Error {response.status_code}: {response.text}"

def call_anthropic(messages: List[Dict[str, str]]) -> str:
    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        return "Anthropic API key not configured"
    endpoint = "https://api.anthropic.com/v1/messages"
    response = requests.post(
        endpoint,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={"model": "claude-3-haiku-20240307", "messages": messages, "max_tokens": 512},
        timeout=15,
    )
    if response.status_code == 200:
        data = response.json()
        return data.get("content", [{}])[0].get("text", "")
    return f"Error {response.status_code}: {response.text}"

def call_gemini(messages: List[Dict[str, str]]) -> str:
    api_key = config.GEMINI_API_KEY
    if not api_key:
        return "Gemini API key not configured"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    response = requests.post(
        endpoint,
        json={"contents": [{"parts": [{"text": m["content"]} for m in messages]}]},
        timeout=15,
    )
    if response.status_code == 200:
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return f"Error {response.status_code}: {response.text}"

# ---- ElevenLabs Text-to-Speech ----

def text_to_speech(text: str) -> bytes:
    api_key = config.ELEVENLABS_API_KEY
    if not api_key:
        return b""
    url = "https://api.elevenlabs.io/v1/text-to-speech/eleven_monolingual_v1"
    response = requests.post(
        url,
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"text": text},
        timeout=15,
    )
    if response.status_code == 200:
        return response.content
    return b""

# ---- Freshdesk Integration ----

def get_freshdesk_tickets() -> List[Dict[str, str]]:
    if not config.FRESHDESK_DOMAIN or not config.FRESHDESK_API_KEY:
        return []
    url = f"https://{config.FRESHDESK_DOMAIN}.freshdesk.com/api/v2/tickets?include=description"
    try:
        response = requests.get(url, auth=(config.FRESHDESK_API_KEY, "X"), timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# ---- Streamlit View ----

def display_ai_solution_center() -> None:
    st.header("AI Solution Center")

    provider = st.selectbox("AI Provider", ["OpenAI", "Anthropic Claude", "Google Gemini"])

    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    user_input = st.text_area("Your message", key="ai_message")
    if st.button("Send", key="send_ai_message") and user_input:
        st.session_state.ai_chat_history.append({"role": "user", "content": user_input})
        if provider == "OpenAI":
            reply = call_openai(st.session_state.ai_chat_history)
        elif provider == "Anthropic Claude":
            reply = call_anthropic(st.session_state.ai_chat_history)
        else:
            reply = call_gemini(st.session_state.ai_chat_history)
        st.session_state.ai_chat_history.append({"role": "assistant", "content": reply})

    for msg in st.session_state.ai_chat_history:
        role = msg["role"]
        if role == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AI:** {msg['content']}")
            audio = text_to_speech(msg["content"])
            if audio:
                st.audio(audio, format="audio/mp3")

    st.markdown("---")
    st.subheader("Freshdesk Tickets")

    # Refresh ticket list every 10 seconds
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10_000, key="freshdesk_refresh")

    if "freshdesk_tickets" not in st.session_state:
        st.session_state.freshdesk_tickets = get_freshdesk_tickets()
    else:
        # Update tickets on each refresh
        st.session_state.freshdesk_tickets = get_freshdesk_tickets()

    tickets = st.session_state.freshdesk_tickets
    if tickets:
        for ticket in tickets:
            with st.expander(f"#{ticket.get('id')} - {ticket.get('subject')}"):
                st.markdown(ticket.get('description_text', 'No description'))
    else:
        st.info("No tickets found or Freshdesk not configured.")
