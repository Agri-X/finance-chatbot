import os
import asyncio
from pathlib import Path
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import chainlit as cl

memory = MemorySaver()


client = MultiServerMCPClient(
    {
        # type: ignore
        "finance": {
            "command": "python",
            "args": [str(Path("mcp-server/investor_agent/server.py").resolve())],
            "env": {
                "PYTHONPATH": str(Path(__file__).resolve().parent),
            },
            "transport": "stdio",
        },
    }
)


tools = asyncio.run(client.get_tools())

model = ChatGoogleGenerativeAI(model=os.getenv("MODEL_BASE", "gemini-2.0-flash"))
final_model = ChatGoogleGenerativeAI(model=os.getenv("MODEL_FINAL", "gemini-2.0-flash"))

model = model.bind_tools(tools)
tool_node = ToolNode(tools=tools)

final_model = final_model.with_config(tags=["final_node"])


def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "final"


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def call_final_model(state: MessagesState):
    messages = state["messages"]
    last_ai_message = messages[-1]
    response = final_model.invoke(
        [
            SystemMessage(
                "you are a Agent to answer user queries regarding finance, generate reports, and provide insights. rewrite the last message to be more concise and clear."
            ),
            HumanMessage(last_ai_message.content),
        ]
    )
    response.id = last_ai_message.id
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("final", call_final_model)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
)
builder.add_edge("tools", "agent")
builder.add_edge("final", END)

graph = builder.compile()


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    async for msg, metadata in graph.astream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "final"
        ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()
