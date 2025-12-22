"""
============================================
INTEGRACIÓN COMPLETA DEL SISTEMA RAG
============================================
Este archivo integra todos los componentes:
1. Pipeline de Recuperación
2. LLM Generator
3. Sistema Conversacional con Memoria

USO EN EL NOTEBOOK:
Ejecuta este código después de tener definidos:
- hybrid_search (búsqueda vectorial)
- table_searcher (búsqueda tabular)
- graph_searcher (búsqueda en grafos)
- KeywordClassifier (clasificador)
"""

# ==================== IMPORTS ====================
# (Asegúrate de haber ejecutado las celdas anteriores con las clases necesarias)

from typing import List, Dict

# ==================== CONFIGURACIÓN ====================

# Primero, asegúrate de tener tu configuración
class Config:
    """Configuración global - Ajusta según tu caso"""
    
    # Elige tu proveedor de LLM
    LLM_PROVIDER = "gemini"  # Opciones: 'gemini', 'openai', 'groq', 'ollama'
    
    # API Keys (comenta las que no uses)
    GEMINI_API_KEY = "tu-api-key-aqui"      # https://aistudio.google.com/apikey
    # OPENAI_API_KEY = "sk-..."             # https://platform.openai.com/api-keys
    # GROQ_API_KEY = "gsk_..."              # https://console.groq.com/keys
    
    # Parámetros del LLM
    LLM_MODEL = "gemini-2.5-flash"  # o "gpt-4o-mini", "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 1024
    
    # Parámetros de recuperación
    RETRIEVAL_TOP_K = 5
    MAX_CONVERSATION_HISTORY = 10

config = Config()

# ==================== PASO 1: CREAR PIPELINE ====================

print("="*60)
print("🔧 INICIALIZANDO SISTEMA RAG")
print("="*60 + "\n")

print("1️⃣ Creando Pipeline de Recuperación...")

# Crear clasificador keyword (backup si el LLM no funciona)
keyword_classifier = KeywordClassifier()

# Crear pipeline de recuperación
pipeline = RetrievalPipeline(
    vector_search=hybrid_search,      # Ya debe estar definido
    table_search=table_searcher,      # Ya debe estar definido
    graph_search=graph_searcher,      # Ya debe estar definido
    classifier=keyword_classifier
)

print("✅ Pipeline creado\n")

# ==================== PASO 2: CREAR LLM GENERATOR ====================

print("2️⃣ Inicializando LLM Generator...")
print(f"   Proveedor: {config.LLM_PROVIDER}")
print(f"   Modelo: {config.LLM_MODEL}")

try:
    if config.LLM_PROVIDER == "gemini":
        llm = create_llm_generator(
            provider='gemini',
            api_key=config.GEMINI_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
    
    elif config.LLM_PROVIDER == "openai":
        llm = create_llm_generator(
            provider='openai',
            api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
    
    elif config.LLM_PROVIDER == "groq":
        llm = create_llm_generator(
            provider='groq',
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
    
    elif config.LLM_PROVIDER == "ollama":
        llm = create_llm_generator(
            provider='ollama',
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE
        )
    
    else:
        raise ValueError(f"Proveedor no soportado: {config.LLM_PROVIDER}")
    
    print("✅ LLM Generator inicializado\n")

except Exception as e:
    print(f"❌ Error al inicializar LLM: {e}")
    print("💡 Verifica tu API key y configuración")
    raise

# ==================== PASO 3: CREAR SISTEMA CONVERSACIONAL ====================

print("3️⃣ Creando Sistema Conversacional...")

rag_system = ConversationalRAG(
    retrieval_pipeline=pipeline,
    llm_generator=llm,
    memory=ConversationMemory(max_history=config.MAX_CONVERSATION_HISTORY),
    language='es'
)

print("✅ Sistema RAG completo inicializado\n")
print("="*60)
print("🎉 SISTEMA LISTO PARA USAR")
print("="*60 + "\n")

# ==================== PRUEBAS RÁPIDAS ====================

print("🧪 EJECUTANDO PRUEBAS RÁPIDAS...\n")

# Definir consultas de prueba
test_queries = [
    "¿Cómo uso mi licuadora para hacer smoothies?",              # Vectorial
    "¿Cuáles son las licuadoras de menos de $200?",             # Tabular
    "¿Qué productos son compatibles con la batidora P0005?",    # Grafo
    "¿Qué opinan los usuarios de las licuadoras TechHome?",     # Vectorial
    "Muestra productos con garantía mayor a 24 meses"           # Tabular
]

# Ejecutar pruebas
results = batch_test(rag_system, test_queries)

# ==================== MODO INTERACTIVO ====================

print("\n" + "="*60)
print("💬 MODO INTERACTIVO")
print("="*60)
print("Para iniciar el chat interactivo, ejecuta:")
print(">>> interactive_chat(rag_system)")
print("\nO prueba consultas individuales:")
print(">>> result = rag_system.chat('¿Cómo funciona mi licuadora?')")
print(">>> print(result['response'])")
print("="*60 + "\n")

# ==================== EJEMPLOS DE USO ====================

def ejemplo_consulta_simple():
    """Ejemplo de consulta simple"""
    print("\n" + "="*60)
    print("EJEMPLO 1: Consulta Simple")
    print("="*60 + "\n")
    
    result = rag_system.chat("¿Cómo limpio mi licuadora?")
    
    print(f"Consulta: {result['query']}")
    print(f"Respuesta: {result['response']}")
    print(f"Fuente: {result['source']}")
    print(f"Documentos recuperados: {result['retrieval_count']}")


def ejemplo_conversacion():
    """Ejemplo de conversación con contexto"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Conversación con Contexto")
    print("="*60 + "\n")
    
    # Resetear conversación
    rag_system.reset_conversation()
    
    # Primera consulta
    r1 = rag_system.chat("¿Qué licuadoras tienes disponibles?")
    print(f"🧑 Usuario: ¿Qué licuadoras tienes disponibles?")
    print(f"🤖 Asistente: {r1['response'][:200]}...\n")
    
    # Segunda consulta (con contexto)
    r2 = rag_system.chat("¿Cuál me recomiendas para hacer smoothies?")
    print(f"🧑 Usuario: ¿Cuál me recomiendas para hacer smoothies?")
    print(f"🤖 Asistente: {r2['response'][:200]}...\n")
    
    # Tercera consulta
    r3 = rag_system.chat("¿Cuánto cuesta?")
    print(f"🧑 Usuario: ¿Cuánto cuesta?")
    print(f"🤖 Asistente: {r3['response'][:200]}...\n")


def ejemplo_fuente_especifica():
    """Ejemplo especificando la fuente de datos"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Especificar Fuente de Datos")
    print("="*60 + "\n")
    
    # Forzar búsqueda vectorial
    r1 = rag_system.chat(
        "información sobre licuadoras",
        source='vectorial',
        top_k=3
    )
    print(f"Búsqueda vectorial: {r1['response'][:150]}...")
    
    # Forzar búsqueda tabular
    r2 = rag_system.chat(
        "información sobre licuadoras",
        source='tabular'
    )
    print(f"Búsqueda tabular: {r2['response'][:150]}...")


# ==================== EJECUTAR EJEMPLOS ====================

# Descomenta para ejecutar ejemplos:
# ejemplo_consulta_simple()
# ejemplo_conversacion()
# ejemplo_fuente_especifica()

# ==================== ESTADÍSTICAS FINALES ====================

def mostrar_estadisticas():
    """Muestra estadísticas del sistema"""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DEL SISTEMA")
    print("="*60)
    
    print(rag_system.get_conversation_summary())
    pipeline.print_stats()
    llm.print_stats()

# Descomentar para ver estadísticas:
# mostrar_estadisticas()

# ==================== NOTAS IMPORTANTES ====================

"""
NOTAS DE USO:

1. CONFIGURACIÓN INICIAL:
   - Ajusta Config() con tu proveedor de LLM preferido
   - Asegúrate de tener las API keys necesarias
   - Verifica que todos los componentes previos estén definidos

2. PROVEEDORES DE LLM RECOMENDADOS:
   
   a) Gemini (Google):
      - Gratuito con límites
      - API: https://aistudio.google.com/apikey
      - Modelo: gemini-2.5-flash
      - Nota: Puede tener filtros de seguridad restrictivos
   
   b) OpenAI:
      - Requiere créditos ($5 mínimo)
      - API: https://platform.openai.com/api-keys
      - Modelo: gpt-4o-mini (más barato y rápido)
      - Muy confiable, excelente calidad
   
   c) Groq:
      - GRATIS con límites generosos
      - API: https://console.groq.com/keys
      - Modelo: llama-3.3-70b-versatile
      - MUY RÁPIDO, excelente opción gratuita
   
   d) Ollama:
      - Completamente local
      - Requiere instalación: https://ollama.ai
      - Modelo: llama3, mistral, etc.
      - Sin costos, sin límites, privado

3. RECOMENDACIÓN PARA EL TP:
   
   Si Gemini no funciona (filtros de seguridad), usa GROQ:
   - Es gratuito
   - Sin filtros restrictivos
   - Muy rápido
   - Excelente calidad de respuestas
   
   Código para cambiar a Groq:
   ```python
   config.LLM_PROVIDER = "groq"
   config.GROQ_API_KEY = "gsk_..."  # Tu API key de Groq
   config.LLM_MODEL = "llama-3.3-70b-versatile"
   ```

4. PARA EL INFORME:
   
   Menciona:
   - Proveedor de LLM usado y justificación
   - Modelo específico
   - Configuración (temperatura, max_tokens)
   - Resultados de las pruebas
   - Limitaciones encontradas

5. TROUBLESHOOTING:
   
   - Error de API key: Verifica que esté bien copiada
   - Respuestas bloqueadas: Cambia de proveedor
   - Errores de rate limit: Espera o cambia de proveedor
   - Respuestas en inglés: Verifica language='es' en ConversationalRAG
"""
