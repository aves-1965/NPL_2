# Sistema RAG para Electrodomésticos - Novogar

Trabajo Práctico Final de NLP para TUIA (Universidad Tecnológica Nacional). Sistema de Retrieval-Augmented Generation (RAG) con agente autónomo ReAct para consultas sobre productos de electrodomésticos.

## Configuración inicial

El archivo `.env` con las credenciales de acceso a GROQ y Neo4j se envía por email junto con el link a este repositorio. Colocar dicho archivo en el directorio de trabajo antes de ejecutar el notebook.

## Ejecución

Abrir `Novogar_v5_4.ipynb` en Google Colab. Subir el archivo `.env` y el archivo `fuentes.zip` con los datos al entorno de Colab. Ejecutar las celdas en orden secuencial.

## Estructura del proyecto

El notebook implementa un sistema RAG completo con tres fuentes de datos: base vectorial (ChromaDB), base tabular (Pandas) y base de grafos (Neo4j). Incluye clasificador de intención, búsqueda híbrida con reranking, y un agente ReAct con cuatro herramientas especializadas.

## Autor

Alfredo Sanz - TUIA 2025
