"""
============================================
FUNCIÓN LLM PARA GENERACIÓN
============================================
Wrapper para llamadas a LLMs con soporte para:
- Gemini (Google)
- OpenAI (GPT)
- Ollama (Local)
- Groq (Gratuito)
"""

from typing import List, Dict, Optional
import time

# ==================== CLASE BASE LLM ====================

class LLMGenerator:
    """Clase base para generación con LLMs"""
    
    def __init__(self, provider: str = "gemini", **kwargs):
        """
        Inicializa el generador LLM
        
        Args:
            provider: Proveedor del LLM ('gemini', 'openai', 'ollama', 'groq')
            **kwargs: Argumentos específicos del proveedor
        """
        self.provider = provider.lower()
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0
        }
        
        # Inicializar según proveedor
        if self.provider == "gemini":
            self._init_gemini(**kwargs)
        elif self.provider == "openai":
            self._init_openai(**kwargs)
        elif self.provider == "ollama":
            self._init_ollama(**kwargs)
        elif self.provider == "groq":
            self._init_groq(**kwargs)
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")
    
    # ==================== GEMINI ====================
    
    def _init_gemini(self, api_key: str, model: str = "gemini-2.5-flash", **kwargs):
        """Inicializa Gemini"""
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                'temperature': kwargs.get('temperature', 0.7),
                'max_output_tokens': kwargs.get('max_tokens', 1024),
                'top_p': kwargs.get('top_p', 0.95),
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        self.model_name = model
    
    def _generate_gemini(self, prompt: str) -> str:
        """Genera respuesta con Gemini"""
        response = self.model.generate_content(prompt)
        
        # Manejar respuesta bloqueada
        if not response.candidates or response.candidates[0].finish_reason != 1:
            raise ValueError("Respuesta bloqueada por filtros de seguridad")
        
        return response.candidates[0].content.parts[0].text
    
    # ==================== OPENAI ====================
    
    def _init_openai(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        """Inicializa OpenAI"""
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 1024)
    
    def _generate_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Genera respuesta con OpenAI"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    # ==================== OLLAMA ====================
    
    def _init_ollama(self, model: str = "llama3", base_url: str = "http://localhost:11434", **kwargs):
        """Inicializa Ollama"""
        import requests
        
        self.model_name = model
        self.base_url = base_url
        self.temperature = kwargs.get('temperature', 0.7)
    
    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Genera respuesta con Ollama"""
        import requests
        
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        return response.json()['response']
    
    # ==================== GROQ ====================
    
    def _init_groq(self, api_key: str, model: str = "llama-3.3-70b-versatile", **kwargs):
        """Inicializa Groq"""
        from groq import Groq
        
        self.client = Groq(api_key=api_key)
        self.model_name = model
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 1024)
    
    def _generate_groq(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Genera respuesta con Groq"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    # ==================== MÉTODO PRINCIPAL ====================
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 2
    ) -> str:
        """
        Genera respuesta con el LLM configurado
        
        Args:
            prompt: Prompt del usuario
            system_prompt: Prompt del sistema (opcional)
            max_retries: Reintentos en caso de error
            
        Returns:
            Respuesta generada
        """
        self.stats['total_calls'] += 1
        
        for attempt in range(max_retries + 1):
            try:
                # Llamar al proveedor apropiado
                if self.provider == "gemini":
                    # Gemini no soporta system prompt separado, combinar
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    response = self._generate_gemini(full_prompt)
                
                elif self.provider == "openai":
                    response = self._generate_openai(prompt, system_prompt)
                
                elif self.provider == "ollama":
                    response = self._generate_ollama(prompt, system_prompt)
                
                elif self.provider == "groq":
                    response = self._generate_groq(prompt, system_prompt)
                
                else:
                    raise ValueError(f"Proveedor no implementado: {self.provider}")
                
                self.stats['successful_calls'] += 1
                return response
            
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Error (intento {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"   Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.stats['failed_calls'] += 1
                    raise Exception(f"Error después de {max_retries + 1} intentos: {e}")
    
    def print_stats(self):
        """Imprime estadísticas de uso"""
        print("\n" + "="*60)
        print(f"📊 ESTADÍSTICAS DEL LLM ({self.provider.upper()})")
        print("="*60)
        print(f"Modelo:            {self.model_name}")
        print(f"Total de llamadas: {self.stats['total_calls']}")
        print(f"  ✅ Exitosas:     {self.stats['successful_calls']}")
        print(f"  ❌ Fallidas:     {self.stats['failed_calls']}")
        print("="*60)


# ==================== FUNCIÓN HELPER ====================

def create_llm_generator(provider: str = "gemini", **kwargs) -> LLMGenerator:
    """
    Factory function para crear generador LLM
    
    Args:
        provider: 'gemini', 'openai', 'ollama', 'groq'
        **kwargs: Argumentos específicos del proveedor
        
    Returns:
        LLMGenerator configurado
        
    Ejemplos:
        # Gemini
        llm = create_llm_generator('gemini', api_key=config.GEMINI_API_KEY)
        
        # OpenAI
        llm = create_llm_generator('openai', api_key=config.OPENAI_API_KEY, model='gpt-4o-mini')
        
        # Groq (gratuito)
        llm = create_llm_generator('groq', api_key=config.GROQ_API_KEY)
        
        # Ollama (local)
        llm = create_llm_generator('ollama', model='llama3')
    """
    return LLMGenerator(provider=provider, **kwargs)
