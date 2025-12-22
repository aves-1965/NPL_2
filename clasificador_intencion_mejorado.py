"""
============================================
CLASIFICADOR DE INTENCIÓN - VERSIÓN MEJORADA
============================================
Determina qué base de datos usar según la consulta del usuario
Dos enfoques: LLM Few-Shot vs Modelo Entrenado
Incluye: Rate Limiting, Manejo de Errores, Reintentos
"""

import numpy as np
from typing import Dict, List, Optional
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import google.generativeai as genai

# ==================== DATASET SINTÉTICO ====================

def create_synthetic_dataset() -> tuple:
    """Crea dataset sintético para entrenamiento y evaluación"""

    # Ejemplos de cada clase
    vectorial_queries = [
        "¿Cómo uso mi licuadora para hacer smoothies?",
        "¿Qué opinan los usuarios de esta cafetera?",
        "Mi licuadora no enciende, ¿qué hago?",
        "¿Cómo limpio mi procesadora?",
        "¿Es normal que mi batidora haga ruido?",
        "Instrucciones para usar la picadora",
        "¿Qué dicen las reseñas del producto P0001?",
        "¿Cómo se mantiene la licuadora?",
        "Problemas comunes con el exprimidor",
        "Manual de uso de la batidora",
        "¿Cómo preparar masa con la procesadora?",
        "Opiniones sobre las licuadoras TechHome",
        "¿Qué hacer si mi producto tiene fugas?",
        "¿Cómo funciona el modo pulse?",
        "Feedback de usuarios sobre cafeteras",
        "¿Es seguro usar la picadora continuamente?",
        "¿Cómo picar vegetales correctamente?",
        "Reseñas de la marca ChefMaster",
        "¿Por qué mi licuadora vibra mucho?",
        "¿Cómo hacer smoothies cremosos?",
    ]

    tabular_queries = [
        "¿Cuáles son las licuadoras de menos de $200?",
        "Mostrar productos de la marca TechHome",
        "¿Qué productos tienen garantía mayor a 24 meses?",
        "Licuadoras con voltaje 220V",
        "Productos en stock",
        "¿Cuánto cuesta la procesadora P0013?",
        "Productos con potencia mayor a 1000W",
        "Filtrar por color rojo",
        "¿Qué hay disponible en menos de $150?",
        "Productos con capacidad de 2 litros",
        "Mostrar batidoras baratas",
        "¿Cuál es el precio de la licuadora compacta?",
        "Productos de la categoría Cocina",
        "Filtrar por marca HomeChef",
        "¿Qué productos cuestan menos de $100?",
        "Stock de productos en sucursal Centro",
        "Productos con 36 meses de garantía",
        "¿Cuántas ventas hubo en noviembre?",
        "Distribución de ventas por método de pago",
        "Top 10 productos más vendidos",
    ]

    grafo_queries = [
        "¿Qué productos son compatibles con P0016?",
        "Productos relacionados con licuadoras",
        "¿Qué accesorios hay para la batidora?",
        "Productos similares al P0001",
        "¿Qué repuestos comparte con otros productos?",
        "¿Dónde hay stock del producto P0005?",
        "Productos de la misma categoría que P0013",
        "¿Qué productos usan el mismo motor?",
        "Accesorios compatibles con la picadora",
        "¿En qué sucursales está disponible P0020?",
        "Productos relacionados en la categoría cocina",
        "¿Qué productos comparten componentes con P0008?",
        "¿Qué marca fabrica productos similares?",
        "Productos compatibles de TechHome",
        "¿Dónde puedo conseguir repuestos?",
        "Productos que comparten cuchillas",
        "¿Qué otros productos de CookElite son compatibles?",
        "Stock por sucursal del producto P0001",
        "Productos relacionados con procesadoras",
        "¿Qué accesorios son intercambiables?",
    ]

    # Crear dataset
    queries = (
        vectorial_queries +
        tabular_queries +
        grafo_queries
    )

    labels = (
        ['vectorial'] * len(vectorial_queries) +
        ['tabular'] * len(tabular_queries) +
        ['grafo'] * len(grafo_queries)
    )

    return queries, labels

# ==================== CLASIFICADOR BASADO EN LLM (FEW-SHOT) CON RATE LIMITING ====================

class LLMClassifier:
    """Clasificador usando Gemini con Few-Shot Learning y Rate Limiting"""

    def __init__(self, api_key: str, requests_per_minute: int = 4):
        """
        Inicializa el clasificador LLM

        Args:
            api_key: API key de Google Gemini
            requests_per_minute: Límite de requests por minuto (default: 4 para seguridad)
        """
        genai.configure(api_key=api_key)

        self.llm = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 50,
            },
            safety_settings={
                'HARASSMENT': 'block_none',
                'HATE_SPEECH': 'block_none',
                'SEXUALLY_EXPLICIT': 'block_none',
                'DANGEROUS_CONTENT': 'block_none',
            }
        )

        # Control de rate limiting
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute  # Tiempo mínimo entre requests
        self.last_request_time = 0

        # Estadísticas
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'safety_blocks': 0,
            'rate_limit_errors': 0,
            'fallback_uses': 0
        }

        self.prompt_template = """Clasifica la siguiente consulta en UNA de estas categorías:

CATEGORÍAS:
- vectorial: Preguntas sobre USO, FUNCIONAMIENTO, PROBLEMAS, MANTENIMIENTO, OPINIONES de productos
- tabular: Consultas de PRECIOS, FILTROS por características, STOCK, ESPECIFICACIONES, VENTAS
- grafo: Productos RELACIONADOS, COMPATIBILIDAD, ACCESORIOS, SIMILARES, STOCK POR SUCURSAL

EJEMPLOS:

Consulta: "¿Cómo uso mi licuadora para hacer smoothies?"
Categoría: vectorial

Consulta: "¿Cuáles son las licuadoras de menos de $200?"
Categoría: tabular

Consulta: "¿Qué productos son compatibles con P0016?"
Categoría: grafo

Consulta: "¿Qué opinan los usuarios de esta cafetera?"
Categoría: vectorial

Consulta: "Productos con voltaje 220V"
Categoría: tabular

Consulta: "¿Dónde hay stock del producto P0001?"
Categoría: grafo

CONSULTA: "{query}"

Responde SOLO con una palabra (vectorial, tabular o grafo):"""

    def _wait_for_rate_limit(self):
        """Espera el tiempo necesario para respetar el rate limit"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_delay:
            wait_time = self.min_delay - time_since_last_request
            print(f"  ⏳ Rate limiting: esperando {wait_time:.1f}s...")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def predict(self, query: str, max_retries: int = 2) -> str:
        """
        Predice la clase de una consulta con reintentos

        Args:
            query: Consulta a clasificar
            max_retries: Número máximo de reintentos en caso de error

        Returns:
            Clase predicha (vectorial, tabular o grafo)
        """
        self.stats['total_requests'] += 1
        prompt = self.prompt_template.format(query=query)

        for attempt in range(max_retries + 1):
            try:
                # Respetar rate limit
                self._wait_for_rate_limit()

                # Generar respuesta
                response = self.llm.generate_content(prompt)
                prediction = response.text.strip().lower()

                # Validar que sea una clase válida
                if prediction in ['vectorial', 'tabular', 'grafo']:
                    self.stats['successful_requests'] += 1
                    return prediction
                else:
                    # Si la respuesta no es válida, usar fallback
                    print(f"  ⚠️  Respuesta inválida del LLM: '{prediction}' -> usando fallback")
                    self.stats['fallback_uses'] += 1
                    return self._keyword_fallback(query)

            except Exception as e:
                error_msg = str(e)

                # Detectar tipo de error
                if 'finish_reason' in error_msg and ('2' in error_msg or 'SAFETY' in error_msg):
                    self.stats['safety_blocks'] += 1
                    print(f"  🛡️  Bloqueo de seguridad (intento {attempt + 1}/{max_retries + 1})")

                elif '429' in error_msg or 'quota' in error_msg.lower():
                    self.stats['rate_limit_errors'] += 1
                    print(f"  ⏱️  Rate limit excedido (intento {attempt + 1}/{max_retries + 1})")

                    # Extraer tiempo de espera si está disponible
                    if 'retry in' in error_msg.lower():
                        try:
                            import re
                            wait_match = re.search(r'retry in ([\d.]+)s', error_msg)
                            if wait_match:
                                wait_time = float(wait_match.group(1))
                                print(f"  ⏳ Esperando {wait_time:.1f}s como sugiere la API...")
                                time.sleep(wait_time + 1)  # Esperar un poco más por seguridad
                                continue
                        except:
                            pass

                    # Espera exponencial
                    if attempt < max_retries:
                        wait_time = 2 ** attempt * 10  # 10s, 20s, 40s...
                        print(f"  ⏳ Esperando {wait_time}s antes de reintentar...")
                        time.sleep(wait_time)
                        continue

                else:
                    print(f"  ❌ Error desconocido: {error_msg[:100]}...")

                # Si es el último intento, usar fallback
                if attempt == max_retries:
                    print(f"  🔄 Máximo de reintentos alcanzado -> usando fallback")
                    self.stats['failed_requests'] += 1
                    self.stats['fallback_uses'] += 1
                    return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> str:
        """Clasificación por palabras clave como fallback"""
        query_lower = query.lower()

        # Palabras clave para cada clase
        vectorial_keywords = ['cómo', 'como', 'usar', 'funciona', 'problema', 'opinión', 'reseña', 'manual', 'limpiar', 'mantener']
        tabular_keywords = ['precio', 'menos de', 'mayor que', 'stock', 'cuánto', 'filtrar', 'ventas', 'garantía', 'voltaje', 'potencia']
        grafo_keywords = ['compatible', 'relacionado', 'similar', 'accesorio', 'repuesto', 'donde hay', 'sucursal', 'comparte']

        # Contar coincidencias
        vectorial_score = sum(1 for kw in vectorial_keywords if kw in query_lower)
        tabular_score = sum(1 for kw in tabular_keywords if kw in query_lower)
        grafo_score = sum(1 for kw in grafo_keywords if kw in query_lower)

        # Retornar la clase con mayor score
        scores = {
            'vectorial': vectorial_score,
            'tabular': tabular_score,
            'grafo': grafo_score
        }

        return max(scores, key=scores.get)

    def predict_batch(self, queries: List[str], show_progress: bool = True) -> List[str]:
        """
        Predice múltiples consultas con progreso

        Args:
            queries: Lista de consultas
            show_progress: Mostrar progreso de clasificación

        Returns:
            Lista de clases predichas
        """
        predictions = []
        total = len(queries)

        print(f"\n🔄 Clasificando {total} consultas...")
        print(f"   Rate limit: {self.requests_per_minute} req/min (~{self.min_delay:.1f}s entre requests)")

        for i, query in enumerate(queries, 1):
            if show_progress:
                print(f"\n[{i}/{total}] Clasificando: '{query[:60]}...'")

            pred = self.predict(query)
            predictions.append(pred)

            if show_progress:
                print(f"   ✅ Clasificado como: {pred}")

        return predictions

    def print_stats(self):
        """Imprime estadísticas de uso del clasificador"""
        print(f"\n{'='*60}")
        print("📊 ESTADÍSTICAS DEL CLASIFICADOR LLM")
        print(f"{'='*60}")
        print(f"Total de requests:       {self.stats['total_requests']}")
        print(f"  ✅ Exitosos:           {self.stats['successful_requests']}")
        print(f"  ❌ Fallidos:           {self.stats['failed_requests']}")
        print(f"  🛡️  Bloqueos seguridad: {self.stats['safety_blocks']}")
        print(f"  ⏱️  Rate limit errors:  {self.stats['rate_limit_errors']}")
        print(f"  🔄 Usos de fallback:   {self.stats['fallback_uses']}")
        print(f"{'='*60}\n")

# ==================== CLASIFICADOR BASADO EN KEYWORDS (BASELINE) ====================

class KeywordClassifier:
    """Clasificador simple basado en palabras clave como baseline"""

    def __init__(self):
        self.vectorial_keywords = ['cómo', 'como', 'usar', 'funciona', 'problema', 'opinión', 'reseña', 'manual', 'limpiar', 'mantener']
        self.tabular_keywords = ['precio', 'menos', 'mayor', 'stock', 'cuánto', 'filtrar', 'ventas', 'garantía', 'voltaje', 'potencia']
        self.grafo_keywords = ['compatible', 'relacionado', 'similar', 'accesorio', 'repuesto', 'donde', 'sucursal', 'comparte']

    def predict(self, query: str) -> str:
        """Predice la clase de una consulta"""
        query_lower = query.lower()

        vectorial_score = sum(1 for kw in self.vectorial_keywords if kw in query_lower)
        tabular_score = sum(1 for kw in self.tabular_keywords if kw in query_lower)
        grafo_score = sum(1 for kw in self.grafo_keywords if kw in query_lower)

        scores = {
            'vectorial': vectorial_score,
            'tabular': tabular_score,
            'grafo': grafo_score
        }

        return max(scores, key=scores.get)

    def predict_batch(self, queries: List[str]) -> List[str]:
        """Predice múltiples consultas"""
        return [self.predict(q) for q in queries]

# ==================== EVALUACIÓN ====================

def evaluate_classifier(classifier, X_test: List[str], y_test: List[str], name: str):
    """Evalúa un clasificador y muestra métricas"""

    print(f"\n{'='*60}")
    print(f"EVALUACIÓN: {name}")
    print(f"{'='*60}\n")

    # Predicciones según tipo de clasificador
    if isinstance(classifier, LLMClassifier):
        y_pred = classifier.predict_batch(X_test, show_progress=True)
        classifier.print_stats()
    else:
        # Para otros clasificadores (como KeywordClassifier)
        y_pred = classifier.predict_batch(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred,
        average='weighted',
        zero_division=0
    )

    print(f"📊 Métricas globales:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

    print(f"\n📋 Reporte por clase:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Ejemplos de predicciones
    print(f"🔍 Ejemplos de predicciones:")
    for i in range(min(5, len(X_test))):
        emoji = "✅" if y_pred[i] == y_test[i] else "❌"
        print(f"  {emoji} '{X_test[i][:60]}...'")
        print(f"     Esperado: {y_test[i]} | Predicho: {y_pred[i]}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }
