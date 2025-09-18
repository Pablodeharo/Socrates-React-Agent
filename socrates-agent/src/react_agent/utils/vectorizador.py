"""Utility & helper functions."""

import psycopg2
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import logging

DB_CONFIG = {
    'dbname': 'socrates_vdb',
    'user': 'postgres',
    'password': 'Pa87*blo',
    'host': 'localhost',
    'port': '5432'
}

# Configuración de logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimensión de all-MiniLM-L6-v2

class VectorizadorPlatonDB:
    def __init__(self, db_config: Dict[str, str], model_name: str = MODEL_NAME):
        """
        Inicializa el vectorizador con configuración de DB y modelo de embeddings.
        """
        self.db_config = db_config
        self.model_name = model_name
        self.model = None 
        self.conn = None
        self.cursor = None

    def conectar_db(self):
        """Conecta a la base de datos PostgreSQL."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logger.info("Conexión establecida con PostgreSQL")
        except Exception as e:
            logger.error(f"Error conectando a la DB: {e}")
            raise

    def cargar_modelo(self):
        """Carga el modelo de embeddings."""
        try:
            logger.info(f"Cargando modelo {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Modelo de embeddings cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise

    def crear_esquema_vectorial(self, schema_sql: str):
        """Crea tablas, índices y funciones en PostgreSQL según el SQL proporcionado."""
        try:
            self.cursor.execute(schema_sql)
            self.conn.commit()
            logger.info("Esquema vectorial creado o actualizado correctamente")
        except Exception as e:
            logger.error(f" Error creando esquema vectorial: {e}")
            self.conn.rollback()
            raise

    # ------------------- Métodos de vectorización -------------------

    def vectorizar_documentos(self, max_chars: int = 8000):
        """Vectoriza todos los documentos y guarda embeddings en la DB."""
        logger.info("Vectorizando documentos...")
        query = "SELECT id, titulo, texto FROM documentos_nlp"
        self.cursor.execute(query)
        documentos = self.cursor.fetchall()
        logger.info(f"{len(documentos)} documentos encontrados")

        for i, (doc_id, titulo, texto) in enumerate(documentos):
            try:
                embedding_titulo = self.model.encode(titulo).tolist()
                embedding_texto = self.model.encode(texto[:max_chars]).tolist()

                insert_sql = """
                    INSERT INTO documento_embeddings 
                    (documento_id, embedding_titulo, embedding_texto, modelo_usado)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (documento_id, modelo_usado) 
                    DO UPDATE SET
                        embedding_titulo = EXCLUDED.embedding_titulo,
                        embedding_texto = EXCLUDED.embedding_texto,
                        fecha_creacion = CURRENT_TIMESTAMP
                """
                self.cursor.execute(insert_sql, (doc_id, embedding_titulo, embedding_texto, self.model_name))

                if (i + 1) % 10 == 0:
                    self.conn.commit()
                    logger.info(f"Procesados {i + 1}/{len(documentos)} documentos")

            except Exception as e:
                logger.error(f"Error procesando documento {doc_id}: {e}")
                continue

        self.conn.commit()
        logger.info("Vectorización de documentos completada")

    def vectorizar_fragmentos(self, max_chunk_size: int = 500):
        """Vectoriza fragmentos de documentos largos."""
        logger.info("Vectorizando fragmentos de documentos largos...")
        query = "SELECT id, texto FROM documentos_nlp WHERE LENGTH(texto) > %s"
        self.cursor.execute(query, (max_chunk_size * 2,))
        documentos_largos = self.cursor.fetchall()

        for doc_id, texto in documentos_largos:
            try:
                fragmentos = self._dividir_texto_inteligente(texto, max_chunk_size)
                for i, frag in enumerate(fragmentos):
                    embedding = self.model.encode(frag).tolist()
                    insert_sql = """
                        INSERT INTO fragmento_embeddings 
                        (documento_id, fragmento_texto, fragmento_num, embedding, num_tokens, modelo_usado)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    self.cursor.execute(insert_sql, (doc_id, frag, i, embedding, len(frag.split()), self.model_name))
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error fragmentando documento {doc_id}: {e}")
                continue

        logger.info("Vectorización de fragmentos completada")

    def vectorizar_conceptos(self):
        """Vectoriza conceptos filosóficos únicos."""
        logger.info("Vectorizando conceptos filosóficos...")
        query = """
            SELECT concepto, COUNT(*) as frecuencia, COUNT(DISTINCT documento_id) as docs,
                   STRING_AGG(contexto, ' | ' ORDER BY LENGTH(contexto) DESC) as contextos
            FROM conceptos_filosoficos
            GROUP BY concepto
            HAVING COUNT(*) >= 2
        """
        self.cursor.execute(query)
        conceptos = self.cursor.fetchall()

        for concepto, frecuencia, docs, contextos in conceptos:
            try:
                contexto_ejemplo = contextos.split(' | ')[0] if contextos else concepto
                texto_embedding = f"{concepto}: {contexto_ejemplo}"
                embedding = self.model.encode(texto_embedding).tolist()
                insert_sql = """
                    INSERT INTO concepto_embeddings
                    (concepto, contexto_ejemplo, embedding, frecuencia_total, documentos_mencionan, modelo_usado)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (concepto, modelo_usado)
                    DO UPDATE SET
                        contexto_ejemplo = EXCLUDED.contexto_ejemplo,
                        embedding = EXCLUDED.embedding,
                        frecuencia_total = EXCLUDED.frecuencia_total,
                        documentos_mencionan = EXCLUDED.documentos_mencionan,
                        fecha_creacion = CURRENT_TIMESTAMP
                """
                self.cursor.execute(insert_sql, (concepto, contexto_ejemplo, embedding, frecuencia, docs, self.model_name))
            except Exception as e:
                logger.error(f"Error procesando concepto '{concepto}': {e}")
                continue

        self.conn.commit()
        logger.info("Vectorización de conceptos completada")

    # ------------------- Métodos auxiliares -------------------
    def _dividir_texto_inteligente(self, texto: str, max_size: int) -> List[str]:
        """Divide texto en fragmentos respetando límites de oración."""
        oraciones = texto.split('. ')
        fragmentos, fragmento_actual = [], ""
        for oracion in oraciones:
            if len(fragmento_actual + oracion) > max_size and fragmento_actual:
                fragmentos.append(fragmento_actual.strip())
                fragmento_actual = oracion + ". "
            else:
                fragmento_actual += oracion + ". "
        if fragmento_actual.strip():
            fragmentos.append(fragmento_actual.strip())
        return fragmentos

    def cerrar_conexion(self):
        """Cierra la conexión de manera segura."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Conexión a PostgreSQL cerrada")

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)
