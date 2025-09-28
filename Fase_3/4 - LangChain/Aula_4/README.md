# Exemplos de Chains no LangChain

Este projeto contém exemplos práticos de diferentes tipos de Chains no LangChain, demonstrando como criar fluxos de trabalho sequenciais e sistemas de roteamento para processamento de linguagem natural.

## Exemplos Disponíveis

### 1. Sequential Chain - Gerador de Rap (`exemplo_1_sequential_chain_rap.py`)

Demonstra o uso de `SequentialChain` para criar um fluxo de trabalho que:
- Gera uma letra de rap baseada em um tema
- Verifica se a letra contém conteúdo inadequado
- Faz uma verificação final com metadados

Conceitos demonstrados:
- Criação de múltiplos prompts em sequência
- Uso de `SimpleMemory` para passar dados adicionais
- Formatação de saída estruturada

### 2. Router Chain - Consultor Financeiro (`exemplo_2_router_chain_financeiro.py`)

Implementa um sistema que roteia perguntas financeiras para diferentes especialistas usando `RouterChain`:
- Especialista em ações
- Especialista em renda fixa

Conceitos demonstrados:
- Criação de múltiplos prompts especializados
- Roteamento baseado no conteúdo
- Uso de tipos estruturados com `TypedDict`
- Composição de chains com operadores

## Configuração

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure sua chave da API OpenAI:**
   - Crie um arquivo `.env` na raiz do projeto
   - Adicione sua chave da API:
     ```
     OPENAI_API_KEY=sua-chave-aqui
     ```

## Como Executar os Exemplos

Cada exemplo pode ser executado diretamente como um script Python:

```bash
# Para o exemplo de Sequential Chain (gerador de rap)
python exemplo_1_sequential_chain_rap.py

# Para o exemplo de Router Chain (consultor financeiro)
python exemplo_2_router_chain_financeiro.py
```

## Estrutura do Projeto

```
.
├── exemplo_1_sequential_chain_rap.py    # Exemplo de Sequential Chain
├── exemplo_2_router_chain_financeiro.py # Exemplo de Router Chain
├── requirements.txt                     # Dependências do projeto
└── README.md                           # Este arquivo
```

## Tipos de Chains Demonstrados

### Sequential Chain
- Execução linear de prompts
- A saída de uma etapa é usada como entrada da próxima
- Útil para processamento em múltiplas etapas
- Permite adicionar metadados através de `SimpleMemory`

### Router Chain
- Roteamento dinâmico baseado no conteúdo
- Direciona consultas para especialistas específicos
- Usa tipos estruturados para garantir consistência
- Demonstra composição moderna de chains com operadores `|`