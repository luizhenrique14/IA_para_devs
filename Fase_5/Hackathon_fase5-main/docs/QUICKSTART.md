# ⚡ Guia de Início Rápido - MVP Completo

## 🎉 Status: Fases 1-5 Implementadas!

### ✅ O que foi implementado:

#### **FASE 1: Setup do Projeto**
- ✅ Estrutura de pastas criada
- ✅ requirements.txt com todas as dependências
- ✅ Arquivo .env.example para configuração da API
- ✅ .gitignore configurado

#### **FASE 2: Base de Conhecimento STRIDE**
- ✅ stride_knowledge.py implementado
- ✅ Definições completas da metodologia STRIDE
- ✅ Mapeamento de ameaças por componente
- ✅ Contramedidas catalogadas

#### **FASE 3: Análise com LLM (GPT-4 Vision)**
- ✅ analyzer.py com integração OpenAI
- ✅ Análise automática de diagramas
- ✅ Geração de relatórios estruturados em JSON
- ✅ Tratamento de erros robusto

#### **FASE 4: API REST com FastAPI**
- ✅ main.py implementado
- ✅ Endpoints para análise de diagramas
- ✅ Validação de arquivos e segurança
- ✅ Documentação automática (Swagger/OpenAPI)
- ✅ CORS configurado

#### **FASE 5: Interface Web com Streamlit**
- ✅ app.py implementado
- ✅ Upload e preview de imagens
- ✅ Análise em tempo real
- ✅ Download de relatórios
- ✅ Interface intuitiva e responsiva

---

## ⚙️ Instalação Rápida

### 1. Configure o Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (macOS/Linux)
source venv/bin/activate

# Ativar (Windows)
# venv\Scripts\activate
```

### 2. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 3. Configure a Chave da API OpenAI

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar e adicionar sua chave
# OPENAI_API_KEY=sk-your-api-key-here
```

> 🔑 **Obter chave**: https://platform.openai.com/api-keys

---

## 🚀 Como Usar o Sistema Completo

### Método 1: Interface Web (Recomendado) 🌐

#### Terminal 1 - Inicie a API:

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Iniciar API
uvicorn main:app --reload
```

✅ API disponível em: http://localhost:8000
📖 Documentação: http://localhost:8000/docs

#### Terminal 2 - Inicie a Interface Web:

```bash
# Em outro terminal, ative o ambiente
source venv/bin/activate

# Iniciar Streamlit
streamlit run app.py
```

✅ Interface disponível em: http://localhost:8501

#### Usar a Interface:

1. Acesse http://localhost:8501
2. Faça upload de um diagrama de arquitetura
3. Clique em "Analisar Ameaças"
4. Revise os resultados
5. Baixe o relatório em JSON

---

### Método 2: API REST Direto 🚀

#### Iniciar API:

```bash
uvicorn main:app --reload
```

#### Testar Endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Analisar imagem
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@examples/arquitetura.png"

# Ver metodologia STRIDE
curl http://localhost:8000/stride-info
```

---

### Método 3: Script Python Direto 🐍

#### Opção A: Via linha de comando

```bash
python analyzer.py examples/sua_imagem.png
```

#### Opção B: Via código Python

Crie um arquivo `test_analyzer.py`:

```python
from analyzer import analyze_architecture
import json

# Testar análise
result = analyze_architecture("examples/arquitetura.png")

if result["success"]:
    print("✅ Análise bem-sucedida!")
    print("\n" + "="*80)
    print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
else:
    print(f"❌ Erro: {result['error']}")
```

Execute:

```bash
python test_analyzer.py
```

---

## 📁 Estrutura do Projeto

```
Hackathon/
├── .env.example                          # Template de configuração
├── .gitignore                            # Arquivos ignorados
├── README.md                             # Documentação completa
├── QUICKSTART.md                         # Guia rápido (este arquivo)
├── PLANO_IMPLEMENTACAO_SIMPLIFICADO.md   # Plano detalhado
├── main.py                               # ⭐ API REST FastAPI (FASE 4)
├── app.py                                # ⭐ Interface Streamlit (FASE 5)
├── analyzer.py                           # ⭐ Analisador principal (FASE 3)
├── stride_knowledge.py                   # ⭐ Base STRIDE (FASE 2)
├── test_analyzer.py                      # Testes
├── requirements.txt                      # Dependências (FASE 1)
└── examples/              # Pasta para imagens de teste
```

---

## 🧪 Exemplo de Resultado

Quando você analisa uma imagem, recebe algo como:

```json
{
  "success": true,
  "analysis": {
    "componentes": [
      {
        "nome": "API Gateway",
        "tipo": "api",
        "ameacas": [
          {
            "categoria_stride": "S",
            "descricao": "Falsificação de tokens JWT",
            "criticidade": "Alta",
            "contramedidas": [
              "Implementar validação robusta de tokens",
              "Usar certificados para assinatura"
            ]
          }
        ]
      }
    ],
    "resumo": {
      "total_componentes": 5,
      "total_ameacas": 15,
      "ameacas_alta": 3
    }
  }
}
```

---

## 🎯 Próximas Etapas (Fases 4 e 5)

### FASE 4: API REST com FastAPI
- [ ] Criar endpoint `/analyze` para upload de imagens
- [ ] Endpoint `/health` para monitoramento
- [ ] Documentação automática com Swagger

### FASE 5: Interface Web com Streamlit
- [ ] Interface para upload de imagens
- [ ] Visualização de resultados
- [ ] Export de relatórios

---

## 💡 Dicas

1. **Teste com imagens de exemplo**: Coloque suas imagens de arquitetura na pasta `examples/`

2. **Formato recomendado**: PNG ou JPEG, resolução mínima 800x600

3. **Consulte a base STRIDE**: Use o módulo `stride_knowledge.py`:
   ```python
   from stride_knowledge import get_stride_info
   info = get_stride_info()
   print(info["methodology"])
   ```

4. **Custos da API**: GPT-4 Vision tem custo por token. Monitore em: https://platform.openai.com/usage

---

## 🐛 Troubleshooting

### Erro: "Module 'openai' not found"
```bash
pip install -r requirements.txt
```

### Erro: "OpenAI API key not found"
- Verifique se o arquivo `.env` existe (copie de `.env.example`)
- Confirme que a chave começa com `sk-`

### Erro: "Image file not found"
- Use caminho absoluto ou relativo correto
- Verifique se o arquivo existe: `ls examples/`

---

## 📞 Suporte

Em caso de dúvidas, consulte:
- [README.md](README.md) - Documentação completa
- [PLANO_IMPLEMENTACAO_SIMPLIFICADO.md](PLANO_IMPLEMENTACAO_SIMPLIFICADO.md) - Plano original

---

**Status Atual**: ✅ Fase 3 Completa - Pronto para implementar API REST (Fase 4)
