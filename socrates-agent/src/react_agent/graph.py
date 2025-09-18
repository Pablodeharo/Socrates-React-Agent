# ReAct Agent - Grafo Principal
# ================================
# 
# Este módulo define un agente ReAct (Reasoning and Action) personalizado que funciona
# con modelos de chat que soportan llamadas a herramientas. El agente utiliza LangGraph
# para crear un flujo de trabajo que combina razonamiento y acción.
#
# **Características principales:**
# - Modelo de lenguaje local (Eva Mistral 7B Spanish)
# - Múltiples herramientas: Wikipedia, TTS, Calculadora, Búsqueda vectorial
# - Flujo ReAct: Razón → Acción → Observación → Razón...
# - Base de datos vectorial para búsqueda semántica de documentos

from langgraph.graph import StateGraph, START, END
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from react_agent.state import AgentState
from react_agent.tools import wikipedia_search, text_to_speech, calculator_node
from react_agent.prompts import get_system_prompt
import json
from dotenv import load_dotenv
from react_agent.utils.vectorizador import DB_CONFIG, VectorizadorPlatonDB
from react_agent.prompts import TOOL_FOLLOWUP
import re, json
import os

# Importación de herramientas de búsqueda vectorial
from react_agent.tools import (
    wikipedia_search,
    text_to_speech,
    calculator_node,
    buscar_documentos_por_contenido,
    buscar_conceptos_relacionados,
    buscar_fragmentos_especificos,
    analizar_contexto_concepto,
    comparar_documentos_por_conceptos
)

# Cargar variables de entorno
load_dotenv()
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# ================================
# CONFIGURACIÓN DEL MODELO LLM
# ================================

# Descarga automática del modelo desde Hugging Face Hub
model_path = hf_hub_download(
    repo_id="ecastera/eva-mistral-7b-spanish-GGUF",
    filename="Turdus-trained-20-int4.gguf"
)

"""
Que hace temperature:
Controla la aleatoriedad / creatividad del modelo.
Valores tipicos:
- 0.0 - 04 -> determinista, coherente, predicible
- 0.5- 0.7 -> balance creativo y coherente, ideal para socrates
- 0.8 - 1.0 muy creativo  
"""

# Instancia del modelo de chat optimizada para conversación socrática
llm_socrates = ChatLlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    n_batch=512,
    max_tokens=218, # 1024
    n_ctx=4096,                 # Contexto máximo del modelo
    temperature=0.4,
    top_p=0.9,                  # Núcleo de probabilidad para diversidad controlada
    repeat_penalty=1.1
)

# ================================
# FUNCIÓN PRINCIPAL DEL LLM
# ================================

def llm_call(state: AgentState):
    """ 
    Función central que maneja la llamada al LLM con el contexto de mensajes
    y parsea la acción si el LLM decide usar una herramienta.
    
    Args:
        state (AgentState): Estado actual del agente con mensajes y contexto
        
    Returns:
        AgentState: Estado actualizado con la respuesta del LLM y acción parseada
    """
    messages = state.get("messages", [])

    # Asegurar que el primer mensaje sea el prompt del sistema
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=get_system_prompt(full=True))] + messages

    print("Messages enviados al LLM:", messages)

    # Invocar el modelo y obtener respuesta
    response = llm_socrates.invoke(messages)
    raw_content = response.content.strip()
    print("Respuesta crdua LLM:", repr(raw_content))

    # Limpiar respuesta de tokens de instrucción residuales
    clean_content = (
        raw_content
        .replace("[INST]", "")
        .replace("[/INST]", "")
        .strip()
    )

    # Fallback si la respuesta está vacía
    if not clean_content:
        clean_content = "El LLM no generó texto"

    # Inicializar variables de acción y entrada de herramienta
    action = None
    tool_input = None

    # Buscar y parsear JSON para acciones de herramientas
    json_match = re.search(r'\{.*\}', clean_content, flags=re.S)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            action = parsed.get("action")
            tool_input = parsed.get("input")
            print("Parsed JSON:", parsed)
        except json.JSONDecodeError:
            print("Encontrado algo con forma de JSON pero no válido:", json_match.group(0))

    # Actualizar estado con la respuesta procesada
    state["messages"] = [AIMessage(content=clean_content)]
    state["action"] = action
    state["tool_input"] = tool_input

    return state

# ================================
# WRAPPERS DE HERRAMIENTAS
# ================================
# Cada wrapper adapta las herramientas individuales para trabajar
# con el estado del agente y proporcionar retroalimentación consistente


def wikipedia_wrapper(state: AgentState):
    """
    Wrapper para búsquedas en Wikipedia.
    Ejecuta la búsqueda y proporciona retroalimentación al agente.
    """
    tool_input = state.get("tool_input", "")
    result = wikipedia_search(tool_input)
    followup = TOOL_FOLLOWUP["wikipedia"]
    return {
        "messages": [AIMessage(content=f"Información de Wikipedia: {result}\n\n{followup}")],
        "action": None,
        "last_tool_used": "wikipedia"
    }

def tts_wrapper(state: AgentState):
    """
    Wrapper para generación de audio (Text-to-Speech).
    Convierte texto a audio y confirma la operación.
    """
    tool_input = state.get("tool_input", "")
    result = text_to_speech(tool_input)
    followup = TOOL_FOLLOWUP["tts"]
    return {
        "messages": [AIMessage(content=f"Audio generado: {result}\n\n{followup}")],
        "action": None,
        "last_tool_used": "tts"
    }

def calculator_wrapper(state: AgentState):
    """
    Wrapper para operaciones matemáticas.
    Ejecuta cálculos y devuelve el resultado formateado.
    """
    tool_input = state.get("tool_input", "")
    calc_state = {"expression": tool_input}
    result_state = calculator_node(calc_state)
    result = result_state.get("result", "Error en cálculo")
    followup = TOOL_FOLLOWUP["calculator"]
    return {
        "messages": [AIMessage(content=f"Resultado del cálculo: {result}\n\n{followup}")],
        "action": None,
        "last_tool_used": "calculator"
    }

def vector_search_wrapper(state: AgentState):
    """
    Wrapper unificado para todas las operaciones de búsqueda vectorial.
    Decide qué función específica usar según la acción definida en el estado.
    
    Funciones disponibles:
    - buscar_documentos_por_contenido: Búsqueda general por similaridad
    - buscar_conceptos_relacionados: Encuentra conceptos relacionados
    - buscar_fragmentos_especificos: Búsqueda de fragmentos específicos
    - analizar_contexto_concepto: Análisis contextual de conceptos
    - comparar_documentos_por_conceptos: Comparación entre documentos
    """
    from react_agent.tools import (
        buscar_documentos_por_contenido,
        buscar_conceptos_relacionados,
        buscar_fragmentos_especificos,
        analizar_contexto_concepto,
        comparar_documentos_por_conceptos
    )

    # Mapeo de acciones a funciones de búsqueda vectorial
    tool_map = {
        "buscar_documentos_por_contenido": buscar_documentos_por_contenido,
        "buscar_conceptos_relacionados": buscar_conceptos_relacionados,
        "buscar_fragmentos_especificos": buscar_fragmentos_especificos,
        "analizar_contexto_concepto": analizar_contexto_concepto,
        "comparar_documentos_por_conceptos": comparar_documentos_por_conceptos
    }

    # Acción y query
    action = state.get("action", "buscar_documentos_por_contenido")
    tool_fn = tool_map.get(action)
    query = state.get("tool_input", "")

    # Validar que se proporcione texto para búsqueda
    if not query:
        state["messages"].append(AIMessage(content="No se recibió texto para búsqueda vectorial."))
        return state

    # Ejecutamos la función correspondiente
    resultados = tool_fn(query)
    if not resultados:
        contenido = "⚠️ No se encontraron resultados relevantes."
    else:
        # Formateamos resultados genéricos según la función
        if isinstance(resultados, list):
            # Intentamos detectar tipo de resultado para mostrar título/texto
            if "titulo" in resultados[0]:
                contenido = "\n".join([f"{r['titulo']}: {r.get('texto_preview','')}" for r in resultados])
            elif "concepto" in resultados[0]:
                contenido = "\n".join([f"{r['concepto']}: similaridad {r['similaridad']}" for r in resultados])
            else:
                contenido = str(resultados)
        else:
            contenido = str(resultados)

    # Actualizamos el estado
    state["messages"].append(AIMessage(content=contenido))
    state["action"] = None
    state["tool_input"] = None
    state["last_tool_used"] = action

    return state

# ================================
# CONSTRUCCIÓN DEL GRAFO
# ================================

# Inicializar el grafo de estado con el esquema AgentState
graph = StateGraph(AgentState)

# Agregar todos los nodos al grafo
graph.add_node("llm_call", llm_call)
graph.add_node("Wikipedia", wikipedia_wrapper)
graph.add_node("Voz", tts_wrapper)
graph.add_node("Calculadora", calculator_wrapper)
graph.add_node("BusquedaVectorial", vector_search_wrapper)

# Establecer punto de entrada del grafo
graph.add_edge(START, "llm_call")

# ================================
# ENRUTADOR CONDICIONAL
# ================================

def tool_router(state: AgentState):
    """
    Función de enrutamiento que decide el siguiente nodo basándose en la acción
    determinada por el LLM. Implementa la lógica de decisión del patrón ReAct.
    
    Returns:
        str: Nombre del próximo nodo a ejecutar o END para terminar
    """
    action = state.get("action", None)

    # Routing logic para cada tipo de herramienta
    if action == "wikipedia":
        return "Wikipedia"
    elif action == "voz":
        return "Voz"
    elif action == "calcular":
        return "Calculadora"
    elif action in [
        "buscar_documentos_por_contenido",
        "buscar_conceptos_relacionados",
        "buscar_fragmentos_especificos",
        "analizar_contexto_concepto",
        "comparar_documentos_por_conceptos"
    ]:
        return "BusquedaVectorial"
    else:
        return END

# Configurar enrutamiento condicional desde llm_call
graph.add_conditional_edges(
    "llm_call",
    tool_router,
    {
        "Wikipedia": "Wikipedia",
        "Voz": "Voz",
        "Calculadora": "Calculadora",
        "BusquedaVectorial": "BusquedaVectorial",
        END: END,
    }
)

# ================================
# FLUJO DE RETROALIMENTACIÓN
# ================================
# Después de usar cualquier herramienta, volver al LLM para procesar
# la respuesta y decidir la siguiente acción (patrón ReAct)

graph.add_edge("Wikipedia", "llm_call")
graph.add_edge("Voz", "llm_call")
graph.add_edge("Calculadora", "llm_call")
graph.add_edge("BusquedaVectorial", "llm_call")

# Compilar
app = graph.compile()