# 🤖 ReAct Agent - Philosophical AI Assistant

🧠 ¿Qué es esto?
Este proyecto implementa un Agente ReAct (Razonamiento y Actuación) usando LangGraph y un modelo LLM local en español. Los agentes ReAct combinan el razonamiento de LLM con la ejecución de acciones, pensando iterativamente, usando herramientas y actuando sobre observaciones para lograr objetivos.
El agente se especializa en conversaciones filosóficas y puede:

💭 Razonar a través de preguntas filosóficas complejas
🔧 Usar múltiples herramientas para recopilar información y realizar acciones
📚 Buscar en una base de datos vectorizada de textos filosóficos
🗣️ Generar respuestas en audio usando síntesis de voz
🧮 Realizar cálculos matemáticos y computaciones de fechas históricas

🛠️ Arquitectura
El sistema sigue el patrón ReAct que permite a los sistemas de IA combinar capacidades de razonamiento de LLMs con ejecución de acciones:
Entrada Usuario → Razonamiento LLM → Selección Herramienta → Ejecución Acción → Observación → Bucle...

Componentes Clave:

graph.py: Implementación del flujo ReAct principal con LangGraph
state.py: Esquema de gestión del estado del agente
tools.py: Colección de herramientas especializadas (Wikipedia, TTS, Calculadora, Búsqueda Vectorial)
prompts.py: Prompts del sistema y guía de herramientas
utils/vectorizador.py: Cliente de base de datos vectorial para textos filosóficos

## 🚀 Instalación
```bash
git clone https://github.com/tuusuario/my-socrates-agent.git
cd react_agent
python -m venv .venv
source .venv/bin/activate
# .venv\Scripts\activate   # en Windows
pip install -r requirements.txt

