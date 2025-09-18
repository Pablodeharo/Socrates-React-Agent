# Herramientas del Agente ReAct
# =============================
#
# Conjunto de herramientas especializadas que el agente puede usar para:
# - Cálculos matemáticos y fechas históricas
# - Conversión de texto a voz (TTS)
# - Búsquedas en Wikipedia
# - Búsqueda semántica en base de datos vectorial de textos filosóficos
#
# Cada herramienta está decorada con @tool para integración con LangChain
# y maneja errores de forma robusta para mantener la estabilidad del agente.

from langchain_core.tools import tool
from sympy import sympify
from datetime import datetime
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write
import numpy as np
import wikipedia
import psycopg2
from sentence_transformers import SentenceTransformer
from react_agent.utils.vectorizador import DB_CONFIG, VectorizadorPlatonDB, MODEL_NAME, logger

# ================================
# HERRAMIENTA DE CÁLCULO
# ================================

@tool
def calculator_node(state: dict) -> dict:
    """
    Nodo de cálculo: resuelve expresiones matemáticas y calcula diferencias de años,
    incluyendo fechas en a.C. y d.C.

    @ state: diccionario con posibles claves:
    - "expression": cadena con operación matemática (ej: "2025 - 399")
    - "year": cadena o entero con año histórico (ej: "399 a.C." o "1200 d.C.")

    Funcionamiento:
    - Si se da una "expression", se evalúa de forma segura con sympy.
    - Si se da un "year", se interpreta correctamente:
        · a.C. → año negativo (ej: 399 a.C. → -399)
        · d.C. o sin sufijo → año positivo normal
    - Se calcula la diferencia entre el año actual y el histórico.

    Return: state actualizado con:
    - "result": resultado numérico o error

    def calculator_node(state: dict) -> dict:
    """
    try:
        if "expression" in state and state["expression"]:
            # Intentar evaluar como expresión matemática
            result = sympify(state["expression"]).evalf()
            state["result"] = float(result)
        else:
            # Si no hay expresión, intentar interpretar como año histórico
            expr = state.get("expression", "")
            current_year = datetime.now().year
            year_str = str(expr).lower()

            # Normalizar año
            if "a.c" in year_str or "ac" in year_str:
                # Ej: "399 a.C." → -399
                year_num = -int("".join([c for c in year_str if c.isdigit()]))
            else:
                # Ej: "1200 d.C." o "1200" o expresión matemática
                try:
                    # Primero intentar como expresión matemática
                    result = sympify(expr).evalf()
                    state["result"] = float(result)
                    return state
                except:
                    # Si falla, interpretar como año
                    year_num = int("".join([c for c in expr if c.isdigit()]))

            # Calcular diferencia con año actual
            state["result"] = current_year - year_num

    except Exception as e:
        state["result"] = f"Error: {e}"

    return state

# ================================
# HERRAMIENTA DE SÍNTESIS DE VOZ
# ================================

@tool
def text_to_speech(text: str) -> str:
    """
    Convierte texto a audio usando Bark TTS con voz en español.
    
    Utiliza el modelo Bark para generar audio natural con entonación
    específica para español (speaker_6) y lo guarda en formato WAV.
    
    Args:
        text (str): Texto a sintetizar en audio
        
    Returns:
        str: Mensaje de confirmación con ruta del archivo o error
        
    Características:
    - Voz en español natural (es_speaker_6)
    - Normalización de audio para evitar distorsión
    - Formato WAV 16-bit para compatibilidad universal
    - Manejo robusto de errores
    """
    if not text:
        return "No hay texto para convertir"
    

    try:
        audio_array = generate_audio(text, history_prompt="es_speaker_6")
        filename = "socrates_bark.wav"
        # Normalizamos para WAV
        audio_array = audio_array / np.max(np.abs(audio_array))
        write(filename, SAMPLE_RATE, (audio_array * 32767).astype(np.int16))
        return f"Audio generado exitosamente: {filename}"
    except Exception as e:
        return f"Error generando audio: {e}"

# ================================
# HERRAMIENTA DE WIKIPEDIA
# ================================

@tool
def wikipedia_search(query: str, sentences: int = 5):
    """
    Búsqueda inteligente en Wikipedia en español con manejo de ambiguedad.
    
    Args:
        query (str): Término o frase a buscar
        sentences (int): Número de oraciones del resumen (por defecto 5)
    
    Returns:
        str: Resumen de Wikipedia, opciones de desambiguación, o error
        
    Características:
    - Búsqueda prioritaria en Wikipedia en español
    - Manejo automático de páginas ambiguas
    - Sugerencias cuando no se encuentran resultados exactos
    - Control de longitud del resumen devuelto
    """
    if not query:
        return "No hay consulta para buscar"
        
    try:
        wikipedia.set_lang("es")
        summary = wikipedia.summary(query, sentences=sentences)
        return summary
    except wikipedia.DisambiguationError as e:
        # Si hay ambiguedad, devuelve las opciones disponibles
        return f"Tu consulta es ambigua. Tal vez quisiste decir: {', '.join(e.options[:5])}"
    except wikipedia.PageError:
        return "No se encontró ningún resultado en Wikipedia."
    except Exception as e:
        return f"Error al buscar en Wikipedia: {str(e)}"
    
# ================================
# CONFIGURACIÓN BÚSQUEDA VECTORIAL
# ================================

model = SentenceTransformer("all-MiniLM-L6-v2")

vectorizador = VectorizadorPlatonDB(DB_CONFIG, MODEL_NAME)

# ================================
# HERRAMIENTAS DE BÚSQUEDA VECTORIAL
# ================================

@tool
def buscar_documentos_por_contenido(consulta: str, limite: int = 5):
    """
    Búsqueda semántica de documentos filosóficos por contenido.
    
    Utiliza embeddings para encontrar documentos similares al texto de consulta,
    permitiendo búsquedas conceptuales más que literales.
    
    Args:
        consulta (str): Texto de búsqueda para comparar semánticamente
        limite (int): Número máximo de resultados (por defecto 5)
    
    Returns:
        list: Lista de diccionarios con:
            - documento_id: ID único del documento
            - titulo: Título del documento
            - tipo: Tipo de documento (diálogo, tratado, etc.)
            - similaridad: Puntuación de similitud semántica (0-1)
            - texto_preview: Vista previa del contenido
    """
    try:
        # Generar embedding vectorial de la consulta
        embedding_consulta = vectorizador.model.encode(consulta).tolist()
        
        # Ejecutar búsqueda vectorial en PostgreSQL con pgvector
        vectorizador.cursor.execute(
            "SELECT * FROM buscar_documentos_similares(%s, %s)", 
            (embedding_consulta, limite)
        )
        resultados = vectorizador.cursor.fetchall()
        
        return [
            {
                "documento_id": r[0],
                "titulo": r[1],
                "tipo": r[2],
                "similaridad": r[3],
                "texto_preview": r[4]
            } for r in resultados
        ]
    except Exception as e:
        logger.error(f"Error buscando documentos: {e}")
        return []
    
@tool
def buscar_conceptos_relacionados(concepto: str, limite: int = 10):
    """
    Encuentra conceptos filosóficos relacionados semánticamente.
    
    Busca en la base de datos vectorial conceptos que sean semánticamente
    similares al concepto proporcionado, útil para explorar ideas relacionadas.
    
    Args:
        concepto (str): Concepto filosófico base para encontrar similares
        limite (int): Número máximo de conceptos relacionados (por defecto 10)
    
    Returns:
        list: Lista de diccionarios con:
            - concepto: Concepto relacionado encontrado
            - similaridad: Grado de similitud semántica
            - frecuencia_total: Frecuencia del concepto en la base de datos
            - contexto_ejemplo: Ejemplo de contexto donde aparece el concepto
    """
    try:
        embedding_consulta = vectorizador.model.encode(concepto).tolist()
        vectorizador.cursor.execute(
            "SELECT * FROM buscar_conceptos_similares(%s, %s)", 
            (embedding_consulta, limite)
        )
        resultados = vectorizador.cursor.fetchall()
        
        return [
            {
                "concepto": r[0],
                "similaridad": r[1],
                "frecuencia_total": r[2],
                "contexto_ejemplo": r[3]
            } for r in resultados
        ]
    except Exception as e:
        logger.error(f"Error buscando conceptos relacionados: {e}")
        return []

@tool
def buscar_fragmentos_especificos(consulta: str, limite: int = 10):
    """
    Búsqueda de fragmentos específicos de texto en documentos filosóficos.
    
    Permite encontrar pasajes específicos que sean semánticamente similares
    al texto de consulta, ideal para citas y referencias precisas.
    
    Args:
        consulta (str): Texto de referencia para encontrar fragmentos similares
        limite (int): Máximo número de fragmentos a devolver (por defecto 10)
    
    Returns:
        list: Lista de diccionarios con:
            - documento_id: ID del documento que contiene el fragmento
            - fragmento_num: Número del fragmento dentro del documento
            - fragmento_texto: Texto completo del fragmento
            - similaridad: Puntuación de similitud semántica
    """
    try:
        embedding_consulta = vectorizador.model.encode(consulta).tolist()
        vectorizador.cursor.execute(
            """
            SELECT documento_id, fragmento_num, fragmento_texto,
                   1 - (embedding <=> %s) AS similaridad
            FROM fragmento_embeddings
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (embedding_consulta, embedding_consulta, limite)
        )
        resultados = vectorizador.cursor.fetchall()
        
        return [
            {
                "documento_id": r[0],
                "fragmento_num": r[1],
                "fragmento_texto": r[2],
                "similaridad": r[3]
            } for r in resultados
        ]
    except Exception as e:
        logger.error(f"Error buscando fragmentos: {e}")
        return []

@tool
def analizar_contexto_concepto(concepto: str):
    """
    Análisis exhaustivo de contextos donde aparece un concepto filosófico.
    
    Proporciona un análisis detallado de cómo y dónde se usa un concepto
    específico en la base de datos, incluyendo estadísticas de uso.
    
    Args:
        concepto (str): Concepto filosófico a analizar en detalle
    
    Returns:
        list: Lista de diccionarios con:
            - contexto_ejemplo: Ejemplo de contexto textual del concepto
            - frecuencia_total: Número total de apariciones
            - documentos_mencionan: Cantidad de documentos que lo mencionan
            
    Útil para:
    - Entender el uso histórico de un concepto
    - Analizar evolución semántica
    - Encontrar definiciones y explicaciones contextuales
    """
    try:
        vectorizador.cursor.execute(
            """
            SELECT contexto_ejemplo, frecuencia_total, documentos_mencionan
            FROM concepto_embeddings
            WHERE concepto = %s
            """,
            (concepto,)
        )
        resultados = vectorizador.cursor.fetchall()
        
        return [
            {
                "contexto_ejemplo": r[0],
                "frecuencia_total": r[1],
                "documentos_mencionan": r[2]
            } for r in resultados
        ]
    except Exception as e:
        logger.error(f"Error analizando contexto de concepto '{concepto}': {e}")
        return []

@tool
def comparar_documentos_por_conceptos(titulo1: str, titulo2: str):
    """
    Comparación conceptual entre dos documentos filosóficos.
    
    Analiza qué conceptos filosóficos comparten dos documentos,
    útil para estudios comparativos y análisis de influencias intelectuales.
    
    Args:
        titulo1 (str): Título del primer documento a comparar
        titulo2 (str): Título del segundo documento a comparar
    
    Returns:
        dict: Diccionario con:
            - conceptos_comunes: Lista de conceptos presentes en ambos documentos
            - cantidad_comunes: Número total de conceptos compartidos
    """
    try:
        vectorizador.cursor.execute(
            "SELECT concepto FROM conceptos_filosoficos WHERE titulo = %s",
            (titulo1,)
        )
        conceptos1 = set(r[0] for r in vectorizador.cursor.fetchall())
        
        vectorizador.cursor.execute(
            "SELECT concepto FROM conceptos_filosoficos WHERE titulo = %s",
            (titulo2,)
        )
        conceptos2 = set(r[0] for r in vectorizador.cursor.fetchall())
        
        comunes = conceptos1.intersection(conceptos2)
        return {
            "conceptos_comunes": list(comunes),
            "cantidad_comunes": len(comunes)
        }
    except Exception as e:
        logger.error(f"Error comparando documentos '{titulo1}' y '{titulo2}': {e}")
        return {}