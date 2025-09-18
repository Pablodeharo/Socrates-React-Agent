# Estado del Agente ReAct
# =======================
# 
# Define la estructura de datos que mantiene el estado del agente durante
# la ejecución del flujo ReAct. Este estado se comparte entre todos los nodos
# del grafo y permite mantener el contexto de la conversación y las acciones.

from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    """
    Estado compartido del agente que mantiene el contexto durante la ejecución.
    
    Attributes:
        messages: Historial de mensajes de la conversación con manejo automático
        action: Acción a ejecutar determinada por el LLM (wikipedia, calcular, etc.)
        tool_input: Entrada/parámetros para la herramienta seleccionada
        last_tool_used: Última herramienta utilizada
    """
    messages: Annotated[list[AnyMessage], add_messages]
    action: Optional[str]
    tool_input: Optional[str]
    last_tool_used: Optional[str] 
