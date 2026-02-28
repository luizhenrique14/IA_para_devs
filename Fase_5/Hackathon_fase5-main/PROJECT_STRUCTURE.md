# 🗂️ Estrutura do Projeto STRIDE Threat Analyzer

Este documento descreve a organização completa dos arquivos e diretórios do projeto.

## 📁 Estrutura de Diretórios

```
STRIDE-Threat-Analyzer/
│
├── 📄 README.md                          # Documentação principal do projeto
├── 📄 requirements.txt                   # Dependências Python
├── 📄 .env.example                       # Exemplo de configuração
├── 📄 .gitignore                         # Arquivos ignorados pelo Git
│
├── 🐍 Código Fonte Principal
│   ├── main.py                          # API FastAPI
│   ├── app.py                           # Interface Web Streamlit
│   ├── analyzer.py                      # Analisador com GPT-4 Vision
│   ├── pdf_generator.py                 # Gerador de relatórios PDF
│   ├── stride_knowledge.py              # Base de conhecimento STRIDE
│   └── test_analyzer.py                 # Script de testes
│
├── 📚 docs/                             # Documentação completa
│   ├── README.md                        # Índice da documentação
│   ├── QUICKSTART.md                    # Guia de início rápido
│   ├── PLANO_IMPLEMENTACAO_SIMPLIFICADO.md
│   ├── AVALIACAO_PROJETO.md            # Relatório de testes
│   ├── IADT - Fase 5 - Hackaton.pdf   # Documento do hackathon
│   └── reports/                        # Relatórios de exemplo
│       ├── README.md                   # Info sobre relatórios
│       ├── .gitkeep                    # Mantém dir no Git
│       ├── test_report.pdf             # Exemplo 1
│       └── novo_test_report.pdf        # Exemplo 2
│
├── 📸 examples/                         # Diagramas de exemplo
│   └── README.md                       # Info sobre exemplos
│
├── 🔧 .venv/                           # Ambiente virtual Python
└── 🗑️  __pycache__/                    # Cache Python (ignorado)
```

## 📋 Descrição dos Arquivos

### Arquivos de Configuração

| Arquivo | Descrição |
|---------|-----------|
| `README.md` | Documentação principal com overview do projeto |
| `requirements.txt` | Lista de dependências Python necessárias |
| `.env.example` | Exemplo de variáveis de ambiente |
| `.env` | Configuração real (não versionado) |
| `.gitignore` | Arquivos e diretórios ignorados pelo Git |

### Código Fonte

| Arquivo | Responsabilidade |
|---------|------------------|
| `main.py` | API REST com FastAPI - endpoints de análise |
| `app.py` | Interface web com Streamlit - UI do usuário |
| `analyzer.py` | Módulo de análise com GPT-4 Vision |
| `pdf_generator.py` | Geração de relatórios PDF profissionais |
| `stride_knowledge.py` | Base de conhecimento da metodologia STRIDE |
| `test_analyzer.py` | Scripts de teste e demonstração |

### Documentação (`docs/`)

| Arquivo | Conteúdo |
|---------|----------|
| `README.md` | Índice completo da documentação |
| `QUICKSTART.md` | Tutorial rápido de uso |
| `PLANO_IMPLEMENTACAO_SIMPLIFICADO.md` | Arquitetura e planejamento |
| `AVALIACAO_PROJETO.md` | Resultados de testes e validações |
| `IADT - Fase 5 - Hackaton.pdf` | Documento oficial do hackathon |

### Relatórios de Exemplo (`docs/reports/`)

Contém exemplos de PDFs gerados pelo sistema demonstrando:
- Formato e estrutura dos relatórios
- Análise STRIDE completa
- Contramedidas e recomendações

### Exemplos (`examples/`)

Diretório para armazenar diagramas de arquitetura de teste:
- Diagramas PNG/JPG/JPEG
- Exemplos de diferentes arquiteturas
- Material para testes e demonstrações

## 🔄 Fluxo de Dados

```
┌─────────────────┐
│  User uploads   │
│    diagram      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   app.py        │  ◄── Interface Streamlit
│  (Streamlit)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   main.py       │  ◄── API REST
│   (FastAPI)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  analyzer.py    │  ◄── Análise com IA
│  (GPT-4 Vision) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ stride_knowledge│  ◄── Base de conhecimento
│      .py        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│pdf_generator.py │  ◄── Geração de PDF
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PDF Report     │  ◄── Relatório final
│   Download      │
└─────────────────┘
```

## 📊 Responsabilidades por Módulo

### `main.py` - API Backend
- ✅ Expõe endpoints REST
- ✅ Valida uploads de imagens
- ✅ Gerencia arquivos temporários
- ✅ Retorna análises em JSON ou PDF
- ✅ Documentação automática (Swagger)

### `app.py` - Interface Frontend
- ✅ Interface web amigável
- ✅ Upload de diagramas
- ✅ Visualização de resultados
- ✅ Download de relatórios
- ✅ Gestão de estado da sessão

### `analyzer.py` - Motor de Análise
- ✅ Integração com OpenAI GPT-4 Vision
- ✅ Processamento de imagens
- ✅ Aplicação da metodologia STRIDE
- ✅ Estruturação de resultados

### `pdf_generator.py` - Gerador de Relatórios
- ✅ Criação de PDFs profissionais
- ✅ Formatação de conteúdo
- ✅ Inclusão de diagramas
- ✅ Tabelas e gráficos
- ✅ Branding e layout

### `stride_knowledge.py` - Base de Conhecimento
- ✅ Definições STRIDE
- ✅ Ameaças por tipo de componente
- ✅ Contramedidas recomendadas
- ✅ Melhores práticas de segurança

## 🚀 Pontos de Entrada

| Comando | Componente | Porta | Descrição |
|---------|-----------|-------|-----------|
| `python3 main.py` | API | 8000 | Inicia servidor FastAPI |
| `streamlit run app.py` | Web UI | 8501 | Inicia interface Streamlit |
| `python3 analyzer.py <imagem>` | CLI | - | Análise direta via terminal |
| `python3 test_analyzer.py` | Testes | - | Executa testes do sistema |

## 📦 Dependências Principais

Definidas em `requirements.txt`:

- **openai** - Integração com GPT-4 Vision
- **fastapi** - Framework web para API
- **uvicorn** - Servidor ASGI
- **streamlit** - Interface web interativa
- **pillow** - Processamento de imagens
- **reportlab** - Geração de PDFs
- **python-dotenv** - Gestão de variáveis de ambiente
- **requests** - Cliente HTTP

## 🔒 Arquivos Sensíveis (Não Versionados)

- `.env` - Contém API keys (nunca commitar!)
- `.venv/` - Ambiente virtual Python
- `__pycache__/` - Cache de bytecode Python
- `docs/reports/*.pdf` - Relatórios de teste

## 📝 Convenções

### Nomenclatura de Arquivos
- Módulos Python: `snake_case.py`
- Documentação: `UPPERCASE.md` ou `CamelCase.md`
- Configuração: `.lowercase` ou `.lowercase.example`

### Estrutura de Código
- Classes: `PascalCase`
- Funções: `snake_case`
- Constantes: `UPPER_SNAKE_CASE`
- Variáveis: `snake_case`

### Documentação
- Docstrings em inglês no código
- Documentação externa em português
- Comentários em português
- README em português com emojis

## 🔄 Versionamento

```
docs/reports/        ← Ignorado (arquivos temporários)
.env                 ← Ignorado (credenciais)
.venv/               ← Ignorado (ambiente local)
__pycache__/         ← Ignorado (cache Python)
*.pyc                ← Ignorado (bytecode)
```

## 📚 Documentação Adicional

Para informações mais detalhadas, consulte:

- [README Principal](../README.md) - Overview do projeto
- [Quick Start](docs/QUICKSTART.md) - Como começar
- [Índice de Documentação](docs/README.md) - Toda documentação
- [Relatórios de Exemplo](docs/reports/README.md) - Exemplos de output

## 🎯 Próximas Expansões

Para expandir o projeto, considere adicionar:

```
├── tests/                    # Testes automatizados
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── scripts/                  # Scripts auxiliares
│   ├── deploy.sh
│   └── backup.sh
│
├── data/                     # Dados persistentes
│   ├── cache/
│   └── history/
│
└── docker/                   # Containerização
    ├── Dockerfile
    └── docker-compose.yml
```

---

**Última atualização**: Fevereiro 2026  
**Versão do projeto**: MVP Fase 5 Completa  
**Status**: ✅ Produção
