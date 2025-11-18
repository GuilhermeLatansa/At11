import torch
import os
import time
from requests.exceptions import JSONDecodeError

# Ajuste de importação robusto para a versão 4.34.0 do transformers
try:
    from transformers.tools import HfAgent
except ImportError:
    from transformers import HfAgent 
    
HF_TOKEN = "xxx" 

# ID do modelo mais robusto e compatível com o HfAgent (versão 4.34.0).
MODEL_ID = "OpenAssistant/oasst-sft-4-pythia-12b"
# URL COMPLETA para a API de Inferência.
AGENT_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
MAX_RETRIES = 3 # Número máximo de tentativas de execução
    
# 1. Configurar o Agente
print("Tentando inicializar o Agente de LLM...")
try:
    # Passamos a URL COMPLETA como o primeiro argumento posicional.
    agent = HfAgent(AGENT_URL, token=HF_TOKEN) 
    
    print(f"✅ Agente Inicializado com sucesso (Modelo: {MODEL_ID}).")
    
except Exception as e:
    print(f"\n❌ Erro CRÍTICO ao inicializar o agente via API.")
    print(f"Detalhe: {e}")
    print("\nVerifique estes pontos: O TOKEN está correto e tem permissão 'Write'?")
    exit()
    
# 2. Executar o Agente com Tentativas (Retry Logic)
pergunta = "Quem venceu a Copa do Mundo de 2022 e qual foi o placar da final?"
print(f"\n🧠 Tarefa do Agente: {pergunta}")

for attempt in range(MAX_RETRIES):
    print(f"Aguarde. Tentativa {attempt + 1}/{MAX_RETRIES}. O Agente está raciocinando e usando suas ferramentas...")
    
    try:
        # Tenta executar o agente
        resultado = agent.run(pergunta)
        
        # Se funcionar, exibe e sai do loop
        print("\n✅ Resposta do Agente:")
        print(resultado)
        break # Sai do loop de tentativas
        
    except (JSONDecodeError, Exception) as e:
        # Captura tanto o erro de JSON (Expecting value) quanto qualquer outro erro.
        print(f"\n❌ Erro na execução da tarefa. Detalhe: {type(e).__name__}: {e}")
        
        if attempt < MAX_RETRIES - 1:
            # Calcula o tempo de espera (Backoff Exponencial: 2s, 4s, 8s, etc.)
            wait_time = 2 ** (attempt + 1)
            print(f"Tentando novamente em {wait_time} segundos...")
            time.sleep(wait_time)
        else:
            print("\n🚨 Todas as tentativas falharam.")
            print("Verifique se seu TOKEN tem permissão 'Write' e tente novamente mais tarde (problema de limite de uso da API gratuita).")