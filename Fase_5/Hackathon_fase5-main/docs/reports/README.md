# 📄 Relatórios de Exemplo

Este diretório contém exemplos de relatórios PDF gerados pelo sistema STRIDE Threat Analyzer.

## 📋 Conteúdo

Os relatórios neste diretório são exemplos gerados durante testes do sistema e demonstram:

- **Formato do relatório** - Estrutura e layout dos PDFs gerados
- **Análise completa** - Como as ameaças são apresentadas
- **Metodologia STRIDE** - Explicação da metodologia no relatório
- **Contramedidas** - Sugestões de segurança por componente

## 🎯 Como Gerar Novos Relatórios

### Via Interface Web (Streamlit)

1. Inicie a interface: `streamlit run app.py`
2. Faça upload de um diagrama de arquitetura
3. Clique em "Analisar Ameaças"
4. Após a análise, clique em "Gerar Relatório PDF"
5. Baixe o PDF gerado

### Via API REST

```bash
curl -X POST "http://localhost:8000/analyze-pdf" \
  -H "accept: application/pdf" \
  -F "image=@caminho/para/diagrama.png" \
  -o "meu_relatorio.pdf"
```

## 📦 Estrutura do Relatório

Os relatórios PDF contêm:

1. **Capa** - Título e informações do relatório
2. **Sumário Executivo** - Métricas e visão geral
3. **Diagrama Analisado** - Imagem do diagrama original
4. **Análise Detalhada** - Ameaças por componente com:
   - Categoria STRIDE
   - Descrição da ameaça
   - Nível de criticidade
   - Contramedidas recomendadas
5. **Fluxos de Dados** - Análise de comunicações entre componentes
6. **Recomendações Gerais** - Boas práticas de segurança
7. **Metodologia STRIDE** - Explicação da metodologia

## 📊 Exemplos Disponíveis

Os relatórios neste diretório são gerados a partir de:
- Diagramas de teste simples
- Arquiteturas de microserviços
- Sistemas web (frontend + backend + database)
- APIs e gateways

## 🔒 Nota de Segurança

Os relatórios de teste são automaticamente ignorados pelo Git (via `.gitignore`) para:
- Não versionar arquivos temporários
- Evitar informações sensíveis no repositório
- Manter o repositório limpo

Apenas mantenha exemplos públicos e sem informações confidenciais neste diretório.

---

Para mais informações, consulte a [documentação principal](../README.md).
