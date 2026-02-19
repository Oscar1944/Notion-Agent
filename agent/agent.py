from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory

import asyncio

class Agent:
    def __init__(self, LLM_Client, MCPToolKit):
        self.llm = LLM_Client
        self.tool = MCPToolKit

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. 
             You should answer the user question.
             You have been provided the conversation history, refer to it if needed.
             You have the access to the tools, use it if needed.
             """),
            MessagesPlaceholder(variable_name="chat_history"), # 記憶存放處
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")  # Agent Thinking
        ])

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.agent = create_tool_calling_agent(self.llm, self.tool, self.prompt)
        self.agent_exe = AgentExecutor(
            agent=self.agent,
            tools=self.tool,
            memory=self.memory,  # 關鍵：將 memory 注入執行器
            verbose=True,   # 顯示思考過程
            handle_parsing_errors=True
        )

    async def chat(self, query):
        res = await self.agent_exe.ainvoke({"input":query})

        return res



# dev-test
async def test():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="AIzaSyASSpK37mLNMmjiHESB1QfvTqmAgwy6770")
    mcp_client = MultiServerMCPClient(
        {
            "Tools": {
                "transport": "http",  # HTTP-based remote server
                # Ensure you start your weather server on port 8000
                "url": "http://localhost:7007/mcp"
            }
        }
    )
    toolkit = await mcp_client.get_tools()  
    agent = Agent(llm, toolkit)

    await agent.chat("what is AI, reply me in 10 worlds")

    res = await agent.chat("what is my last question")

    print(res["output"])

# dev-test
# asyncio.run(test())

