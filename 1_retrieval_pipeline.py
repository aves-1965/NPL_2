"""
============================================
PIPELINE DE RECUPERACIÓN (RETRIEVAL)
============================================
Integra todas las fuentes de datos:
- Base Vectorial (con búsqueda híbrida y ReRank)
- Base Tabular (con queries dinámicas)
- Base de Grafos (con queries Cypher dinámicas)
"""

from typing import Dict, List, Optional, Any
import json

# ==================== PIPELINE DE RECUPERACIÓN ====================

class RetrievalPipeline:
    """Pipeline que orquesta la recuperación de información de todas las fuentes"""
    
    def __init__(
        self,
        vector_search,      # HybridSearch ya implementado
        table_search,       # TableSearch ya implementado  
        graph_search,       # GraphSearch ya implementado
        classifier=None     # Clasificador de intención (opcional)
    ):
        """
        Inicializa el pipeline de recuperación
        
        Args:
            vector_search: Sistema de búsqueda vectorial/híbrida
            table_search: Sistema de búsqueda en datos tabulares
            graph_search: Sistema de búsqueda en grafos
            classifier: Clasificador de intención (si existe)
        """
        self.vector_search = vector_search
        self.table_search = table_search
        self.graph_search = graph_search
        self.classifier = classifier
        
        # Estadísticas de uso
        self.stats = {
            'vectorial': 0,
            'tabular': 0,
            'grafo': 0,
            'multi': 0
        }
    
    def retrieve(
        self,
        query: str,
        source: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Recupera información relevante para una consulta
        
        Args:
            query: Consulta del usuario
            source: Fuente específica ('vectorial', 'tabular', 'grafo') o None para auto-detectar
            top_k: Número de resultados a recuperar
            filters: Filtros adicionales (para búsqueda vectorial)
            
        Returns:
            Diccionario con resultados y metadatos
        """
        
        # Auto-detectar fuente si no se especifica
        if source is None:
            source = self._classify_query(query)
        
        # Recuperar de la fuente apropiada
        if source == 'vectorial':
            return self._retrieve_vectorial(query, top_k, filters)
        
        elif source == 'tabular':
            return self._retrieve_tabular(query)
        
        elif source == 'grafo':
            return self._retrieve_grafo(query)
        
        else:
            # Si no se puede determinar, intentar búsqueda múltiple
            return self._retrieve_multi(query, top_k, filters)
    
    def _classify_query(self, query: str) -> str:
        """Clasifica la consulta para determinar la fuente apropiada"""
        # Si hay clasificador, usarlo
        if self.classifier is not None:
            try:
                return self.classifier.predict(query)
            except:
                pass
        
        # Fallback: clasificación por palabras clave
        query_lower = query.lower()
        
        vectorial_kw = ['cómo', 'como', 'usar', 'funciona', 'problema', 'opinión', 'reseña']
        tabular_kw = ['precio', 'cuánto', 'stock', 'filtrar', 'ventas', 'garantía']
        grafo_kw = ['compatible', 'relacionado', 'similar', 'accesorio', 'sucursal']
        
        vec_score = sum(1 for kw in vectorial_kw if kw in query_lower)
        tab_score = sum(1 for kw in tabular_kw if kw in query_lower)
        graph_score = sum(1 for kw in grafo_kw if kw in query_lower)
        
        scores = {'vectorial': vec_score, 'tabular': tab_score, 'grafo': graph_score}
        return max(scores, key=scores.get)
    
    def _retrieve_vectorial(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Recupera de la base vectorial"""
        self.stats['vectorial'] += 1
        results = self.vector_search.search(query=query, top_k=top_k, filters=filters)
        
        return {
            'source': 'vectorial',
            'query': query,
            'results': results,
            'count': len(results),
            'method': 'hybrid_search_with_rerank'
        }
    
    def _retrieve_tabular(self, query: str) -> Dict[str, Any]:
        """Recupera de la base tabular"""
        self.stats['tabular'] += 1
        results = self.table_search.search(query)
        
        return {
            'source': 'tabular',
            'query': query,
            'results': results,
            'count': len(results) if isinstance(results, list) else 1,
            'method': 'dynamic_query'
        }
    
    def _retrieve_grafo(self, query: str) -> Dict[str, Any]:
        """Recupera de la base de grafos"""
        self.stats['grafo'] += 1
        results = self.graph_search.search(query)
        
        return {
            'source': 'grafo',
            'query': query,
            'results': results,
            'count': len(results),
            'method': 'dynamic_cypher_query'
        }
    
    def _retrieve_multi(self, query: str, top_k: int = 3, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Recupera de múltiples fuentes"""
        self.stats['multi'] += 1
        
        results = {'source': 'multi', 'query': query, 'vectorial': None, 'tabular': None, 'grafo': None}
        
        try:
            results['vectorial'] = self.vector_search.search(query, top_k=top_k, filters=filters)
        except Exception as e:
            print(f"⚠️  Error vectorial: {e}")
        
        try:
            results['tabular'] = self.table_search.search(query)
        except Exception as e:
            print(f"⚠️  Error tabular: {e}")
        
        try:
            results['grafo'] = self.graph_search.search(query)
        except Exception as e:
            print(f"⚠️  Error grafo: {e}")
        
        return results
    
    def print_stats(self):
        """Imprime estadísticas"""
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS DEL PIPELINE")
        print("="*60)
        print(f"Vectoriales: {self.stats['vectorial']}")
        print(f"Tabulares:   {self.stats['tabular']}")
        print(f"Grafos:      {self.stats['grafo']}")
        print(f"Múltiples:   {self.stats['multi']}")
        print(f"Total:       {sum(self.stats.values())}")
        print("="*60)


# ==================== FORMATEO DE CONTEXTO ====================

def format_context(retrieval_result: Dict[str, Any]) -> str:
    """Formatea resultados de recuperación como contexto para el LLM"""
    source = retrieval_result.get('source', 'unknown')
    
    if source == 'vectorial':
        results = retrieval_result.get('results', [])
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get('text', result.get('document', 'N/A'))
            metadata = result.get('metadata', {})
            score = result.get('score', 0.0)
            context_parts.append(
                f"[Doc {i}] (Score: {score:.2f})\n"
                f"Fuente: {metadata.get('source', 'N/A')}\n"
                f"{text}\n"
            )
        return "\n".join(context_parts)
    
    elif source == 'tabular':
        results = retrieval_result.get('results', [])
        if isinstance(results, list) and len(results) > 0:
            context_parts = []
            for i, item in enumerate(results, 1):
                item_str = "\n".join([f"  {k}: {v}" for k, v in item.items()])
                context_parts.append(f"[Resultado {i}]\n{item_str}")
            return "\n\n".join(context_parts)
        return str(results)
    
    elif source == 'grafo':
        results = retrieval_result.get('results', [])
        return "\n".join([f"[Relación {i}] {r}" for i, r in enumerate(results, 1)])
    
    elif source == 'multi':
        parts = []
        if retrieval_result.get('vectorial'):
            parts.append("=== DOCUMENTOS ===")
            parts.append(format_context({'source': 'vectorial', 'results': retrieval_result['vectorial']}))
        if retrieval_result.get('tabular'):
            parts.append("\n=== DATOS ===")
            parts.append(format_context({'source': 'tabular', 'results': retrieval_result['tabular']}))
        if retrieval_result.get('grafo'):
            parts.append("\n=== RELACIONES ===")
            parts.append(format_context({'source': 'grafo', 'results': retrieval_result['grafo']}))
        return "\n".join(parts)
    
    return str(retrieval_result)
