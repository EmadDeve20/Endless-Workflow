import asyncio
import os

from typing import Literal, TypedDict

from pydantic import Field

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langgraph.graph import StateGraph, MessagesState, START, END

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import SystemMessage

from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()


class State(MessagesState):
    keep_going: Literal["yes", "no"]


class Route(TypedDict):
    keep_going: Literal["yes", "no"] = Field(..., description="The next step in the routing process"
                                             "no mean turn off your self and yes means you want to keep going. choice it!")

rate_limiter = InMemoryRateLimiter(
    requests_per_second=float(os.getenv("REQUESTS_PER_SECOND")),  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=float(os.getenv("CHECK_EVERY_N_SECONDS")),  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=int(os.getenv("MAX_BUCKET_SIZE")),  # Controls the maximum burst size.
)

llm = HuggingFaceEndpoint(
    model=os.getenv("LLM_MODEL"),
    temperature=float(os.getenv("TEMPERATURE")),
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API"),

)

model = ChatHuggingFace(llm=llm, rate_limiter=rate_limiter)

router = model.with_structured_output(Route)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            os.getenv("SYSTEM_PROMPT"),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

async def think_node(state:State):

    messages = await prompt_template.ainvoke(state["messages"])

    return {
        "messages": await model.ainvoke(messages)
    } 


async def print_thinking_node(state: State):
    print(state["messages"][-1].content)


async def choice_keep_going_node(state:State):
    
    choice = await router.ainvoke(
        [
            SystemMessage(content="do you wana turn off your self or you wana keep going?"
                          "yes means keep going and no means turn of your self"),
            *state["messages"]
        ]
    )

    return {"keep_going": choice["keep_going"]}

async def choice_route(state:State):

    return state["keep_going"] 


graph = StateGraph(State)

graph.add_node("think", think_node)
graph.add_node("print_thinking", print_thinking_node)
graph.add_node("choice_keep_going", choice_keep_going_node)

graph.add_conditional_edges(
    "choice_keep_going",
    choice_route,
    {
        "yes": "think",
        "no": END
    }
)

graph.add_edge(START, "think")
graph.add_edge("think", "print_thinking")
graph.add_edge("print_thinking", "choice_keep_going")

memory = InMemorySaver()

compiled_graph = graph.compile(checkpointer=memory)



async def main():
    config = {"configurable": {"thread_id": "AI-Memory"}}

    message = await compiled_graph.ainvoke({}, config=config)

    print(message)
 

asyncio.run(main())

