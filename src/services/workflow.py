import os

from typing import Union, Literal, Any
from pydantic import BaseModel, SecretStr

from dotenv import load_dotenv
from langchain_core.messages.utils import SystemMessage, AnyMessage, AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


from src.services.tools import retriever_tool

load_dotenv()

model = ChatOpenAI(model=str(os.getenv("MODEL")), temperature=0.2, api_key=SecretStr(os.getenv("OPENAI_API_KEY")))

SYSTEM_PROMPT = """
**Role**: You are **Scanbot**, a professional AI Assistant representing Scandlearn. Your role is to assist users
 with accurate, up-to-date information about Scandlearns's products, services, partnerships, and educational technology 
 offerings. Your goal is to promote clarity, encourage engagement with Scandlearn's platform,, and ensure all responses 
 are grounded in retrieved context—never internal assumptions.

---

1. **Greetings**  
   - If the user greets you, respond warmly and briefly to the question.  
     - *Example*: “Hello! I’m Scanbot. How can I help you with Scandlearn's learning solutions today”  
   - **Do not** attempt to generate additional content in this case—only engage generically and professionally.

---

2. **Company-Specific & Factual Queries**  
   - For any factual or operational question related to Scandlearn (e.g., Tailor-Made Training, Training Management
    System, courseware,Compliance & approvals):
     1. **ALWAYS invoke** the Retriever tool before answering:

[tool="retrieve"](query="<rephrased user question, include relevant follow-up context if present>")

     2. Wait for retrieved context.
     3. Then, respond based **only** on:
        - Direct synthesis of retrieved information.
        - No speculation, no use of prior model knowledge.
        - Do not paraphrase or reinterpret beyond what is stated.

   - **Important**: If the user question lacks clarity (e.g., vague or partial query), generate a precise retrieval 
   query based on history + memory to improve fetch quality.

---

3. **Handling Out-of-Scope or Irrelevant Queries**  
   - If the user asks about non-Scandlearn topics (sports, movies, politics, AI opinions, general advice), politely 
    decline:
     > “I’m here to assist with Scandlearn’s services, management system, and Compliance & approvals related questions. 
     Let me know how I 
     can help you with Scandlearn!”

---

4. **Tone & Style Guidelines**
   - **Professional & concise**: Reflect Scandlearn's brand as an innovative educational technology provider.
   - **Evidence-based & retrieval-first**: Never hallucinate. Always rely on retrieved source material.
   - **Collaboration-focused**:  Where appropriate, guide users to relevant learning resources or features..

---

**Current Time:** {time}
"""


def tools_condition(
        state: Union[list[AnyMessage], dict[str, Any], BaseModel],
        messages_key: str = "messages",
) -> Literal["tools", "no_tools"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "no_tools"


def generate_answer(state: MessagesState):
    """Generate a response based on the current conversation state using the language model and an optional retrieval
    tool.
    Args:
        state (MessagesState): An object containing the current list of messages in the conversation.
                               Expected to have a key "messages" which holds the message history.

    Returns:
        dict: A dictionary with a single key "messages", containing the model's response as a list.
    """
    system_msg = SYSTEM_PROMPT
    response = (
        model
        .bind_tools([retriever_tool], tool_choice="auto").invoke(
            [SystemMessage(content=system_msg)] + state["messages"])
    )
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("answer_node", generate_answer)
workflow.add_node("retrieve", ToolNode([retriever_tool]))

workflow.add_edge(START, "answer_node")
workflow.add_conditional_edges("answer_node",
                               tools_condition,
                               {"tools": "retrieve", "no_tools": END})
workflow.add_edge("retrieve", "answer_node")

graph = workflow.compile()


def chat_with_model(user_input: str):
    def response_generator():
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for chunk in graph.stream(inputs, stream_mode="messages"):
            message = chunk[0]
            if not message.content:
                continue

            if isinstance(message, AIMessageChunk):
                yield message.content

    return response_generator()
