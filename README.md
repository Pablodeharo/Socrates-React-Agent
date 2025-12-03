🤖 ReAct Agent - Philosophical AI Assistant

This project implements a ReAct Agent (Reasoning and Acting) using LangGraph and a local Spanish-language LLM. ReAct agents combine LLM reasoning with tool execution, thinking iteratively, using tools, and acting upon observations to achieve objectives. The agent specializes in philosophical conversations and can:

💭 Reason through complex philosophical questions
🔧 Use multiple tools to gather information and perform actions
📚 Search in a vectorized database of philosophical texts
🗣️ Generate responses in audio using speech synthesis
🧮 Perform mathematical calculations and historical date computations

🛠️ Architecture

The system follows the ReAct pattern, which allows AI systems to combine LLM reasoning capabilities with action execution:

User Input → LLM Reasoning → Tool Selection → Action Execution → Observation → Loop...

Key Components:

graph.py: Implementation of the main ReAct flow using LangGraph

state.py: Agent state management schema

tools.py: Collection of specialized tools (Wikipedia, TTS, Calculator, Vector Search)

prompts.py: System prompts and tool guidance

utils/vectorizador.py: Vector database client for philosophical texts
