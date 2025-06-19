import streamlit as st

from src.services.workflow import chat_with_model

# Streamlit UI
st.set_page_config(page_title="Scand Bot", layout="wide")
st.title("ScandBot - AI Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about Scandlearn...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display AI response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        # Stream response
        for chunk in chat_with_model(user_input):
            full_response += chunk
            response_container.markdown(full_response)

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
