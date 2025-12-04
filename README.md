🤖 ReAct Agent – Philosophical AI Assistant

This project implements a ReAct Agent (Reasoning + Acting) built with LangGraph, LangChain, ChromaDB, BarkAI, and a local Spanish LLM fine-tuned on the complete works of Plato.
The agent engages in philosophical dialogue using a Socratic, reflective reasoning style, combining LLM thinking with tool execution.

✨ Key Features
🧠 Philosophical Intelligence

Generates thoughtful, Socratic-style responses

Encourages reflection through guided questioning

Built on a custom model trained on Plato’s dialogues

🔧 ReAct Agent Loop

The agent behaves in an iterative reasoning format:

Think → Choose a Tool → Act → Observe → Repeat

It can:

Perform step-by-step reasoning

Select and execute tools

Retrieve observations and continue reasoning

🛠️ Integrated Tools

ChromaDB Vector Search → retrieve philosophical texts

Wikipedia API → external factual lookup

BarkAI (Text-to-Speech) → generate spoken responses

Math / Date Calculator → operations and historical date logic

Local Hugging Face Model loaded via pipeline

🏛️ System Architecture

The agent follows the ReAct pattern:

User Input → LLM Reasoning → Tool Call → Action → Observation → Loop

Project Structure
File	Description
graph.py	Main ReAct flow implemented with LangGraph
state.py	Global agent state schema
tools.py	Tool implementations (Wikipedia, TTS, Vector Search, etc.)
prompts.py	System prompts and tool-guidance instructions
configuration.py	Setup of tools, model, and graph
utils/vectorizador.py	Vector database logic for philosophical texts
