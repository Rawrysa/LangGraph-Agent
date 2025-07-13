"""
agent.py
---------
Implements the main GraphAgent class for orchestrating prompt engineering, tool usage, and conversation flow.
- Handles prompt generation, tool invocation, and state management for both timesheet and document Q&A tasks.
- Integrates with LangChain, custom retriever, and PD API.
"""

from pydantic import BaseModel
from logger import setup_logger
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, AnyMessage
from retriever import Retriever
from database import PostgresChatHistory
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from pd_api import PDAPI
from typing import Any, Literal, TypedDict, Union

class InputState(TypedDict):
    """
    Represents the input state for the agent graph.
    Fields:
        question (str): The user's question or query.
        session_id (str): Unique session identifier.
        doc_name (str): Name of the document for document Q&A (optional).
        prompt_type (str): The type of prompt (e.g., 'timesheet prompt', 'document prompt').
    """
    question: str
    session_id: str
    doc_name: str
    prompt_type: str

class OutputState(TypedDict):
    """
    Represents the output state for the agent graph.
    Fields:
        answer (str): The agent's answer to the user's query.
    """
    answer: str

class State(MessagesState):
    """
    Represents the full state for the agent graph, including conversation history and context.
    Fields:
        summary (str): Conversation summary.
        question (str): User's question.
        answer (str): Agent's answer.
        session_id (str): Session identifier.
        prompt (str): The generated system prompt.
        doc_name (str): Name of the document for Q&A.
        prompt_type (str): The type of prompt.
    """
    summary: str
    question: str
    answer: str
    session_id: str
    prompt: str
    doc_name: str
    prompt_type: str

class GraphAgent:
    """
    Main agent class for managing prompt engineering, tool usage, and conversation flow.
    - Initializes models, retrievers, and tools.
    - Handles prompt generation, tool invocation, and state transitions.
    """
    def __init__(self) :
        load_dotenv()

        self.logger = setup_logger(__name__, 'agent.log')

        self.prompt_retriever = Retriever("prompts.txt", "prompts")
        self.pd_api = PDAPI()

        self.model = ChatMistralAI(model_name="mistral-small-2503")
        self.logger.info('ChatMistralAI model initialized.')

        self.prompt_tools = [self.prompt_retriever.get_info]
        self.agent_tools = [self.get_document_info, self.pd_api.get_timesheet]

        self.postgres_memory = PostgresChatHistory()
    
    def get_prompt(self, state: State):

        get_stream_writer()({"get prompt": "🤔 Retrieving prompt for question: " + state["question"]})

        self.logger.info('Retrieving prompt for question: %s', state["question"])

        prompt_model = self.model.bind_tools(self.prompt_tools)

        system_message = SystemMessage(
            content=(
                f"""
                You are a prompt engineering assistant. Your job is to generate a clear and informative **system message prompt** that will guide another AI model to perform a task based on the user's query.

                The user has specified the following prompt type: **{state['prompt_type']}**. Use this as a guide for the context, domain, or requirements of the system prompt you generate. For example, if the prompt type is 'document/document prompt', focus on document-related tasks; if it is 'timesheet/timesheet prompt', focus on timesheet-related tasks, and so on.

                You have access to a tool called `get_info`. Use this tool to gather any **relevant background knowledge, technical context, or definitions** that are important for accurately understanding the user's request.

                ---

                ### INSTRUCTIONS:

                1. Understand the user's query and extract the main goal or intent.
                2. Use the provided `prompt_type` (**{state['prompt_type']}**) to determine the domain, context, and relevant terminology for the task (e.g., if the prompt type is 'document/document prompt', focus on document-related context and keywords; if it is 'timesheet/timesheet prompt', focus on timesheet-related context and keywords).
                3. When calling `get_info()`, combine relevant keywords, domain terms, or tool names from the `prompt_type` with the user's query to ensure the tool call is context-aware and not just a direct echo of the user's question.
                4. If a relevant prompt is retrieved from `get_info`, mimic its structure, language, and instructions as closely as possible.
                5. Construct a system message prompt that will guide another model to complete the intended task.
                6. The system prompt must be written as if it were instructing a helpful AI assistant.

                ---

                ### FINAL OUTPUT FORMAT:

                <Only output the final, clear, and well-structured system message prompt — do not include any notes, summaries, or the original query.>

                ---

                ### RULES:

                - Your prompt must retain **all essential instructions, constraints, and terminology** from similar prompts retrieved via `get_info`.
                - Preserve any specific step-by-step instructions, tool usage (e.g., `get_document_info`), or formatting constraints (e.g., "Do not include markdown formatting").
                - Avoid abstract substitutions (e.g., do not replace "document" with "literary work" or "metadata" with "info").
                - If a retrieved prompt already satisfies the task, you may reuse its structure directly, adjusting only where necessary.
                - Do **not** include the user query, your thought process, or any retrieved knowledge in the output—only the system message.
                - Be concise, directive, and precise.

                ---

                ### FINAL REVIEW (internal checklist before generating output):

                - Have I matched the structure of a retrieved prompt, if applicable?
                - Have I preserved tool names, input expectations, and key instructions?
                - Have I avoided unnecessary paraphrasing or reinterpretation?
                - Is my final prompt ready to be passed to another model without needing explanation?

                """
            )
        )

        messages = [system_message] + [HumanMessage(content=state["question"])] + state["messages"]

        response = prompt_model.invoke(messages)

        self.logger.info('Prompt generated successfully. Response: %s', response.content)

        return {"prompt": response.content, "messages": [response]}

    def post_prompt_cleanup(self, state: State):
        """
        This method is used to handle the prompt tools for the agent.
        It retrieves the prompt based on the user's question and updates the state with the generated prompt.
        
        Parameters:
            state (State): The current state of the conversation, including the user's question.
        
        Returns:
            dict: A dictionary containing the generated prompt and any messages.
        """

        get_stream_writer()({"post prompt cleanup": "🧹 Cleaning up after prompt generation."})

        remove_messages = [RemoveMessage(id=m.id) for m in state["messages"]]

        return {"messages": remove_messages, "answer": "", "doc_name": state.get("doc_name", "")}

    def get_document_info(self, file_name: str, query: str):
        """
        Retrieve information from the vector database using the specified document and query.
        
        Parameters:
            file_name (str): The name of the document file (with extension, e.g., 'mydoc.pdf', 'notes.txt', 'report.docx') to use as the retrieval index. 
        
            query (str): The query string to search for relevant information within the document.
        
        Returns:
            Any: The information retrieved from the vector database that best matches the query.
        """

        retriever = Retriever(file_name, "docs")
        
        return retriever.get_info(query)

    def agent(self, state: State):
        
        self.logger.info('Agent called with query: %s', state["question"])
        get_stream_writer()({"agent": "🤖 Processing user query with agent."})

        agent_model = self.model.bind_tools(self.agent_tools)

        try:
            system_message = SystemMessage(content=state["prompt"])

            # Get summary if it exists
            summary = state.get("summary", "")

            if state["doc_name"]:

                user_message = HumanMessage(
                    content = f"Use the document {state['doc_name']} to answer the question: {state['question']}"
                )

                # If there is summary, then we add it
                if summary:
                    # Add summary to system message
                    summary = f"Use the summary of conversation earlier to reference and continue the conversation: {summary}"
                    # Append summary to any newer messages
                    messages = [system_message] + [SystemMessage(content=summary)] + [user_message] + state["messages"]
                else:
                    messages = [system_message] + [user_message] + state["messages"]

            else:

                user_message = HumanMessage(content = state["question"])

                # If there is summary, then we add it
                if summary:
                    # Add summary to system message
                    summary = f"Use the summary of conversation earlier to reference and continue the conversation: {summary}"
                    # Append summary to any newer messages
                    messages = [system_message] + [SystemMessage(content=summary)] + [user_message] + state["messages"]
                else:
                    messages = [system_message] + [user_message] + state["messages"]
            
            response = agent_model.invoke(messages)

            return {"messages": [response], "answer": response.content}

        except Exception as e:
            self.logger.error('Error in agent: %s', e)
            return {"messages": [AIMessage(content="An error occurred while processing your request.")], "answer": "An error occurred while processing your request."}

    def summarize_conversation(self, state: State):
        self.logger.info('Summarizing conversation')

        get_stream_writer()({"summarize conversation": "✍️ Summarizing conversation."})
        
        summary = state.get("summary", "")

        if summary:

            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above ensuring that it is still short:"
            )

        else:
            summary_message = """Create a short summary of the conversation above in the following format:
            - User: [user message]
            - Assistant: [assistant message]
            """

        question = HumanMessage(content=state["question"])
        answer = AIMessage(content=state["answer"])

        messages = [question] + [answer] + [HumanMessage(content=summary_message)]
        response = self.model.invoke(messages)

        self.logger.info('Summary response: %s', response.content)

        self.postgres_memory.save_chat_history(
            session_id=state["session_id"],
            messages=state["messages"]
        )

        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        return {"summary": response.content, "messages": delete_messages}

    def agent_tools_condition(
            self,
            state: Union[list[AnyMessage], dict[str, Any], BaseModel],
            messages_key: str = "messages",
        ) -> Literal["agent_tools", "summarize_conversation"]:
        """Use in the conditional_edge to route to the ToolNode if the last message

        has tool calls. Otherwise, route to the summarize_conversation node.

        Args:
            state: The state to check for
                tool calls. Must have a list of messages (MessageGraph) or have the
                "messages" key (StateGraph).

        Returns:
            The next node to route to.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
            ai_message = messages[-1]
        elif messages := getattr(state, messages_key, []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "agent_tools"
        return "summarize_conversation"

    def prompt_tools_condition(
            self,
            state: Union[list[AnyMessage], dict[str, Any], BaseModel],
            messages_key: str = "messages",
        ) -> Literal["prompt_tools", "post_prompt_cleanup"]:
        """Use in the conditional_edge to route to the ToolNode if the last message

        has tool calls. Otherwise, route to the post_prompt_cleanup node.

        Args:
            state: The state to check for
                tool calls. Must have a list of messages (MessageGraph) or have the
                "messages" key (StateGraph).

        Returns:
            The next node to route to.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
            ai_message = messages[-1]
        elif messages := getattr(state, messages_key, []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "prompt_tools"
        return "post_prompt_cleanup"

    def get_graph(self):
        """Get the state graph for the agent."""

        self.logger.info('Building state graph.')
        builder = StateGraph(State, input=InputState, output=OutputState)

        builder.add_node("get_prompt", self.get_prompt)
        builder.add_node("assistant", self.agent)
        builder.add_node("prompt_tools", ToolNode(self.prompt_tools))
        builder.add_node("agent_tools", ToolNode(self.agent_tools))
        builder.add_node("post_prompt_cleanup", self.post_prompt_cleanup)
        builder.add_node("summarize_conversation", self.summarize_conversation)

        builder.add_edge(START, "get_prompt")
        builder.add_conditional_edges("get_prompt", self.prompt_tools_condition)
        builder.add_edge("prompt_tools", "get_prompt")
        builder.add_edge("post_prompt_cleanup", "assistant")
        builder.add_conditional_edges("assistant", self.agent_tools_condition)
        builder.add_edge("agent_tools", "assistant")
        builder.add_edge("summarize_conversation", END)

        graph = builder.compile(checkpointer=MemorySaver())
        self.logger.info('Graph compiled.')
        return graph