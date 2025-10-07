# Agente de Análise de Ações com LangChain

Este projeto implementa um agente de análise de ações usando LangChain e OpenAI. O agente pode buscar dados de ações em um banco de dados PostgreSQL e fornecer análises detalhadas usando o modelo de linguagem da OpenAI.

## Estrutura do Projeto

```
stock_agent/
├── sql/
│   ├── schema.sql       # Esquema do banco de dados
│   └── sample_data.sql  # Dados de exemplo
├── database.py          # Módulo de conexão com o banco de dados
├── stock_operations.py  # Operações relacionadas a ações
├── stock_agent.py       # Lógica principal do agente
├── requirements.txt     # Dependências do projeto
├── .env.example        # Exemplo de configuração de variáveis de ambiente
└── README.md           # Este arquivo
```

## Configuração

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure o banco de dados:**
   - Crie um banco de dados PostgreSQL
   - Execute os scripts SQL na seguinte ordem:
     ```bash
     psql -d seu_banco -f sql/schema.sql
     psql -d seu_banco -f sql/sample_data.sql
     ```

3. **Configure as variáveis de ambiente:**
   - Copie o arquivo `.env.example` para `.env`
   - Preencha as variáveis com suas configurações:
     ```
     OPENAI_API_KEY=sua-chave-api-openai
     DB_NAME=seu_banco
     DB_USER=seu_usuario
     DB_PASSWORD=sua_senha
     DB_PORT=5432
     ```

## Uso

Execute o agente principal:

```bash
python stock_agent.py
```

O programa irá:
1. Solicitar o nome ou ticker da ação
2. Buscar dados relevantes no banco de dados
3. Gerar uma análise usando a OpenAI
4. Apresentar os resultados

## Funcionalidades

- **Consulta de Ações**: Busca por nome ou ticker
- **Análise de Dados**: Fornece insights sobre preços e volumes
- **Processamento de Linguagem Natural**: Usa OpenAI para análises detalhadas
- **Interface Interativa**: Modo de console amigável

## Requisitos

- Python 3.8+
- PostgreSQL
- Chave de API da OpenAI
- Dependências listadas em `requirements.txt`

## Desenvolvimento

O código está organizado em módulos:

- `database.py`: Gerenciamento de conexões com o banco de dados
- `stock_operations.py`: Operações específicas de ações
- `stock_agent.py`: Lógica principal do agente e interface com o usuário

## Contribuição

1. Fork o repositório
2. Crie sua branch de feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request