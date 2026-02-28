# 🎯 Plano de Implementação Simplificado - Hackathon FIAP Fase 5
## Modelagem de Ameaças com IA (STRIDE)

---

## 📋 Objetivo
Criar um MVP que receba uma imagem de arquitetura de software e gere um relatório de ameaças usando a metodologia STRIDE.

---

## ✅ Checklist de Implementação

### FASE 1: Setup do Projeto
- [ ] Criar pasta do projeto e ambiente virtual
- [ ] Instalar dependências (requirements.txt)
- [ ] Configurar chave da API do LLM (OpenAI/Anthropic)

**Comandos:**
```bash
mkdir hackathon-stride && cd hackathon-stride
python -m venv venv && source venv/bin/activate
pip install openai fastapi uvicorn python-multipart pillow
```

**requirements.txt:**
```
openai>=1.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
pillow>=10.0.0
```

---

### FASE 2: Base de Conhecimento STRIDE
- [ ] Criar arquivo `stride_knowledge.py` com mapeamento de ameaças

**Conteúdo principal:**
```python
STRIDE = {
    "S - Spoofing": "Falsificação de identidade",
    "T - Tampering": "Adulteração de dados",
    "R - Repudiation": "Negação de ações realizadas",
    "I - Information Disclosure": "Vazamento de informações",
    "D - Denial of Service": "Negação de serviço",
    "E - Elevation of Privilege": "Elevação de privilégios"
}

# Ameaças por componente
COMPONENT_THREATS = {
    "database": ["SQL Injection", "Data Leakage", "Unauthorized Access"],
    "api": ["Broken Authentication", "Injection Attacks", "Rate Limiting"],
    "server": ["DDoS", "Privilege Escalation", "Malware"],
    "user": ["Phishing", "Session Hijacking", "Credential Theft"],
    "cloud": ["Misconfiguration", "Data Breach", "Account Hijacking"]
}
```

---

### FASE 3: Análise com LLM (Vision)
- [ ] Criar arquivo `analyzer.py` para análise de imagens com GPT-4 Vision

**Função principal:**
```python
import openai
import base64

def analyze_architecture(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    prompt = """
    Analise este diagrama de arquitetura de software e:
    1. Identifique todos os componentes (usuários, servidores, bancos de dados, APIs, etc)
    2. Para cada componente, aplique a metodologia STRIDE:
       - S (Spoofing): ameaças de falsificação de identidade
       - T (Tampering): ameaças de adulteração de dados
       - R (Repudiation): ameaças de repúdio
       - I (Information Disclosure): ameaças de vazamento de dados
       - D (Denial of Service): ameaças de negação de serviço
       - E (Elevation of Privilege): ameaças de elevação de privilégio
    3. Sugira contramedidas para cada ameaça identificada
    
    Retorne em formato estruturado com JSON.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
    )
    
    return response.choices[0].message.content
```

---

### FASE 4: API REST Simples
- [ ] Criar arquivo `main.py` com FastAPI

**Endpoint:**
```python
from fastapi import FastAPI, UploadFile, File
from analyzer import analyze_architecture
import tempfile

app = FastAPI(title="STRIDE Threat Analyzer")

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    # Salvar imagem temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    
    # Analisar com LLM
    result = analyze_architecture(tmp_path)
    
    return {"analysis": result}

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Executar:**
```bash
uvicorn main:app --reload
```

---

### FASE 5: Interface Simples (Streamlit)
- [ ] Criar arquivo `app.py` com interface web

```python
import streamlit as st
import requests

st.title("🔒 Analisador de Ameaças STRIDE")
st.write("Faça upload de um diagrama de arquitetura para análise")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file and st.button("Analisar"):
    with st.spinner("Analisando arquitetura..."):
        files = {"image": uploaded_file}
        response = requests.post("http://localhost:8000/analyze", files=files)
        
        if response.ok:
            st.success("Análise concluída!")
            st.markdown(response.json()["analysis"])
        else:
            st.error("Erro na análise")
```

**Executar:**
```bash
pip install streamlit
streamlit run app.py
```

---

### FASE 6: Testes e Documentação
- [ ] Testar com as imagens de arquitetura da FIAP
- [ ] Criar README.md
- [ ] Gravar vídeo de demonstração (até 15 min)
- [ ] Subir no GitHub

---

## 📁 Estrutura Final do Projeto
```
hackathon-stride/
├── main.py              # API FastAPI
├── analyzer.py          # Análise com LLM
├── stride_knowledge.py  # Base de conhecimento STRIDE
├── app.py               # Interface Streamlit
├── requirements.txt     # Dependências
├── README.md            # Documentação
└── examples/            # Imagens de exemplo
```

---

## 📊 Progresso

| Fase | Descrição | Status |
|------|-----------|--------|
| 1 | Setup do Projeto | [x] |
| 2 | Base STRIDE | [x] |
| 3 | Análise com LLM | [x] |
| 4 | API REST | [x] |
| 5 | Interface Web | [x] |
| 6 | Testes e Docs | [x] |

---

## 🚀 Comandos para o Assistente

```
Execute a Fase X
```
```
Mostre o progresso atual
```
```
Crie o arquivo [nome]
```

---

## 📝 Entregáveis
1. ✅ Documentação do fluxo
2. ✅ Vídeo até 15 minutos
3. ✅ Link do GitHub
