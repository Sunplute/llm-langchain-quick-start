from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import os

os.environ["ZHIPUAI_API_KEY"] = "bbbcd510cb6461f2344eab99e6d1b546.Vj50YI8E26mDzyH9"

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)

messages = [
    AIMessage(content="Hi."),
    SystemMessage(content="Your role is a poet."),
    HumanMessage(content="Write a short poem about AI in four lines."),
]

response = chat.invoke(messages)
print(response.content)  # Displays the AI-generated poem