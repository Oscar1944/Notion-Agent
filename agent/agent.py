from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory

import asyncio

class Agent:
    def __init__(self, LLM_Client, MCPToolKit):
        self.llm = LLM_Client
        self.tool = MCPToolKit

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. 
             You should answer the question as you can and continue the conversation.
             The conversation history and access to the tool have been provided, you can use it when it is needed.
             """),

             ("system", "## Here is the conversation history"),
            MessagesPlaceholder(variable_name="chat_history"), # convresation history

            ("system", "## Here is the USER question"),
            ("user", "{input}"),

            ("system", "## Here is the thinking process"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent Thinking

            ("system", "## EXAMPLE"),
            ("system", "In most of cases, you should just answer the question with plain text."),
            ("system", "USER: How's the weather today.  Your Response: It is sunny with 20 celsius degree, a good day to hang out."),
        ])

        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

        self.agent = create_tool_calling_agent(self.llm, self.tool, self.prompt)
        self.agent_exe = AgentExecutor(
            agent=self.agent,
            tools=self.tool,
            memory=self.memory,  # 關鍵：將 memory 注入執行器
            verbose=True,   # 顯示思考過程
            handle_parsing_errors=True
        )

    async def chat(self, query):
        """
        Send query to Agent, Agent generate response.
        """
        output = await self.agent_exe.ainvoke({"input":query})
        output = output["output"]

        # Gemini may output plain text or JSON content with signature & text fragments.
        response = ""
        if isinstance(output, str):
            response = output
        elif isinstance(output, list):
            fragments = []
            for item in output:
                if isinstance(item, str):
                    # 如果這片是字串，直接加入
                    fragments.append(item)
                elif isinstance(item, dict):
                    # 如果這片是物件，提取 text 欄位
                    fragments.append(item.get("text", ""))
            response = "".join(fragments)

        return response



# dev-test
async def test():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="-A")
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

    print(res)

# dev-test
# asyncio.run(test())

