"""
============================================
SISTEMA CONVERSACIONAL CON MEMORIA
============================================
Integra:
- Pipeline de Recuperación
- LLM Generator
- Memoria Conversacional
- Generación de respuestas contextualizadas
"""

from typing import List, Dict, Optional, Any
from datetime import datetime

# ==================== MEMORIA CONVERSACIONAL ====================

class ConversationMemory:
    """Gestiona el historial de conversación"""
    
    def __init__(self, max_history: int = 10):
        """
        Inicializa la memoria conversacional
        
        Args:
            max_history: Máximo de intercambios a recordar
        """
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
        self.metadata: Dict[str, Any] = {
            'start_time': datetime.now(),
            'total_turns': 0
        }
    
    def add_turn(self, user_query: str, assistant_response: str):
        """Agrega un turno de conversación"""
        self.history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now().isoformat()
        })
        self.history.append({
            'role': 'assistant',
            'content': assistant_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limitar tamaño del historial
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
        
        self.metadata['total_turns'] += 1
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Obtiene el historial (completo o últimos N turnos)"""
        if last_n is None:
            return self.history
        return self.history[-(last_n * 2):]
    
    def format_history(self, last_n: Optional[int] = None) -> str:
        """Formatea el historial como texto"""
        history = self.get_history(last_n)
        
        formatted = []
        for msg in history:
            role = "Usuario" if msg['role'] == 'user' else "Asistente"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def clear(self):
        """Limpia el historial"""
        self.history = []
        self.metadata['total_turns'] = 0
        self.metadata['start_time'] = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la conversación"""
        return {
            'total_turns': self.metadata['total_turns'],
            'duration': (datetime.now() - self.metadata['start_time']).total_seconds(),
            'messages': len(self.history)
        }


# ==================== SISTEMA RAG CONVERSACIONAL ====================

class ConversationalRAG:
    """Sistema RAG con capacidades conversacionales"""
    
    def __init__(
        self,
        retrieval_pipeline,
        llm_generator,
        memory: Optional[ConversationMemory] = None,
        language: str = "es"
    ):
        """
        Inicializa el sistema RAG conversacional
        
        Args:
            retrieval_pipeline: Pipeline de recuperación
            llm_generator: Generador LLM
            memory: Memoria conversacional (se crea si no se provee)
            language: Idioma de respuesta ('es' o 'en')
        """
        self.pipeline = retrieval_pipeline
        self.llm = llm_generator
        self.memory = memory or ConversationMemory()
        self.language = language
        
        # System prompt base
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Crea el system prompt base"""
        if self.language == "es":
            return """Eres un asistente virtual especializado en electrodomésticos.

Tu función es ayudar a los clientes con:
- Información sobre productos (uso, características, mantenimiento)
- Consultas de precios y disponibilidad
- Recomendaciones de productos
- Resolución de problemas

INSTRUCCIONES IMPORTANTES:
1. Responde SIEMPRE en español
2. Basa tus respuestas en el CONTEXTO proporcionado
3. Si la información no está en el contexto, indícalo claramente
4. Sé conciso pero completo
5. Sé amigable y profesional
6. Si no entiendes la consulta, pide aclaración

FORMATO DE RESPUESTA:
- Párrafos cortos y claros
- Usa listas cuando sea apropiado
- Menciona fuentes cuando sea relevante
"""
        else:
            return """You are a virtual assistant specialized in home appliances.

Your function is to help customers with:
- Product information (usage, features, maintenance)
- Price and availability queries
- Product recommendations
- Problem solving

IMPORTANT INSTRUCTIONS:
1. Base your answers on the provided CONTEXT
2. If information is not in the context, clearly indicate it
3. Be concise but complete
4. Be friendly and professional
5. If you don't understand the query, ask for clarification
"""
    
    def chat(
        self,
        user_query: str,
        source: Optional[str] = None,
        top_k: int = 5,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa una consulta del usuario y genera respuesta
        
        Args:
            user_query: Consulta del usuario
            source: Fuente específica o None para auto-detectar
            top_k: Número de resultados a recuperar
            include_history: Si incluir historial en el contexto
            
        Returns:
            Diccionario con respuesta y metadatos
        """
        
        # 1. Recuperar información relevante
        retrieval_result = self.pipeline.retrieve(
            query=user_query,
            source=source,
            top_k=top_k
        )
        
        # 2. Formatear contexto
        from retrieval_pipeline import format_context
        context = format_context(retrieval_result)
        
        # 3. Construir prompt
        prompt = self._build_prompt(user_query, context, include_history)
        
        # 4. Generar respuesta
        try:
            response = self.llm.generate(prompt, self.system_prompt)
        except Exception as e:
            response = f"Lo siento, ocurrió un error al procesar tu consulta. Por favor, reformula tu pregunta o intenta nuevamente."
            print(f"❌ Error en LLM: {e}")
        
        # 5. Guardar en memoria
        self.memory.add_turn(user_query, response)
        
        # 6. Retornar resultado completo
        return {
            'query': user_query,
            'response': response,
            'source': retrieval_result.get('source'),
            'context': context,
            'retrieval_count': retrieval_result.get('count', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_prompt(self, query: str, context: str, include_history: bool) -> str:
        """Construye el prompt completo para el LLM"""
        
        parts = []
        
        # Historial (si se solicita)
        if include_history and len(self.memory.history) > 0:
            history = self.memory.format_history(last_n=3)
            parts.append(f"HISTORIAL DE CONVERSACIÓN:\n{history}\n")
        
        # Contexto recuperado
        if context:
            parts.append(f"CONTEXTO RELEVANTE:\n{context}\n")
        
        # Consulta actual
        parts.append(f"CONSULTA ACTUAL:\n{query}\n")
        
        # Instrucción final
        if self.language == "es":
            parts.append("RESPUESTA (en español):")
        else:
            parts.append("RESPONSE:")
        
        return "\n".join(parts)
    
    def get_conversation_summary(self) -> str:
        """Obtiene un resumen de la conversación"""
        stats = self.memory.get_stats()
        
        return f"""
📊 RESUMEN DE LA CONVERSACIÓN
{'='*50}
Turnos totales:  {stats['total_turns']}
Mensajes:        {stats['messages']}
Duración:        {stats['duration']:.1f}s

Uso del Pipeline:
{'-'*50}
"""
    
    def reset_conversation(self):
        """Reinicia la conversación"""
        self.memory.clear()
        print("✅ Conversación reiniciada")


# ==================== INTERFAZ INTERACTIVA ====================

def interactive_chat(rag_system: ConversationalRAG):
    """
    Interfaz de chat interactiva
    
    Args:
        rag_system: Sistema RAG conversacional
    """
    print("\n" + "="*60)
    print("🤖 ASISTENTE VIRTUAL DE ELECTRODOMÉSTICOS")
    print("="*60)
    print("Escribe 'salir' para terminar")
    print("Escribe 'reset' para reiniciar la conversación")
    print("Escribe 'stats' para ver estadísticas")
    print("="*60 + "\n")
    
    while True:
        # Obtener consulta
        user_input = input("🧑 Usuario: ").strip()
        
        # Comandos especiales
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego!")
            print(rag_system.get_conversation_summary())
            rag_system.pipeline.print_stats()
            rag_system.llm.print_stats()
            break
        
        if user_input.lower() == 'reset':
            rag_system.reset_conversation()
            continue
        
        if user_input.lower() == 'stats':
            print(rag_system.get_conversation_summary())
            rag_system.pipeline.print_stats()
            continue
        
        if not user_input:
            continue
        
        # Procesar consulta
        print("\n⏳ Procesando...")
        result = rag_system.chat(user_input)
        
        # Mostrar respuesta
        print(f"\n🤖 Asistente: {result['response']}\n")
        print(f"   📂 Fuente: {result['source']} | Docs: {result['retrieval_count']}")
        print("-" * 60 + "\n")


# ==================== PRUEBAS BATCH ====================

def batch_test(rag_system: ConversationalRAG, queries: List[str]):
    """
    Ejecuta pruebas en lote
    
    Args:
        rag_system: Sistema RAG
        queries: Lista de consultas de prueba
    """
    print("\n" + "="*60)
    print("🧪 PRUEBAS EN LOTE")
    print("="*60 + "\n")
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query}")
        print("-" * 60)
        
        result = rag_system.chat(query)
        results.append(result)
        
        print(f"Respuesta: {result['response'][:200]}...")
        print(f"Fuente: {result['source']} | Docs: {result['retrieval_count']}\n")
    
    print("="*60)
    print("✅ PRUEBAS COMPLETADAS")
    print("="*60)
    
    rag_system.pipeline.print_stats()
    rag_system.llm.print_stats()
    
    return results


# ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    """
    # Ejemplo de uso en el notebook:
    
    # 1. Crear pipeline
    pipeline = RetrievalPipeline(
        vector_search=hybrid_search,
        table_search=table_searcher,
        graph_search=graph_searcher,
        classifier=keyword_classifier
    )
    
    # 2. Crear LLM
    llm = create_llm_generator(
        provider='gemini',
        api_key=config.GEMINI_API_KEY,
        temperature=0.7
    )
    
    # 3. Crear sistema RAG
    rag = ConversationalRAG(
        retrieval_pipeline=pipeline,
        llm_generator=llm,
        language='es'
    )
    
    # 4. Modo interactivo
    interactive_chat(rag)
    
    # O modo batch
    test_queries = [
        "¿Cómo uso mi licuadora?",
        "¿Cuánto cuesta la licuadora compacta?",
        "¿Qué productos son compatibles con la batidora?"
    ]
    batch_test(rag, test_queries)
    """
    pass
