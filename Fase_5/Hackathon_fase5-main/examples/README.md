# 📁 Pasta de Exemplos

Esta pasta deve conter as imagens de arquitetura de software que você deseja analisar usando a metodologia STRIDE.

## 📸 Como Usar

1. **Adicione suas imagens aqui**: Coloque diagramas de arquitetura de software nesta pasta
2. **Formatos aceitos**: PNG, JPG, JPEG
3. **Resolução recomendada**: Mínimo 800x600 pixels para melhor reconhecimento

## 🎯 Exemplos de Diagramas

O analisador funciona melhor com diagramas que incluem:

- ✅ **Componentes claramente identificados**: APIs, servidores, bancos de dados, usuários
- ✅ **Conexões/fluxos visíveis**: Setas mostrando comunicação entre componentes
- ✅ **Rótulos legíveis**: Nomes dos componentes e serviços
- ✅ **Tecnologias mencionadas**: AWS, Azure, Docker, Kubernetes, etc.

## 🔍 Tipos de Arquitetura Suportados

- Arquitetura de Microserviços
- Arquitetura Cliente-Servidor
- Arquitetura em Nuvem (AWS, Azure, GCP)
- Arquitetura Web (Frontend + Backend + DB)
- Arquitetura de APIs
- Sistemas Distribuídos

## 📝 Exemplo de Nomes de Arquivo

```
examples/
├── arquitetura_microservicos.png
├── sistema_web_3camadas.jpg
├── api_gateway_aws.png
└── app_mobile_backend.jpeg
```

## 🚀 Como Analisar

### Método 1: Via Script de Teste
```bash
python test_analyzer.py
```

### Método 2: Análise Direta
```bash
python analyzer.py examples/sua_imagem.png
```

### Método 3: Via Código
```python
from analyzer import analyze_architecture

result = analyze_architecture("examples/arquitetura.png")
if result["success"]:
    print(result["analysis"])
```

## 💡 Dicas para Melhores Resultados

1. **Diagramas bem estruturados**: Use ferramentas como draw.io, Lucidchart, ou AWS Architecture Icons
2. **Contraste adequado**: Fundos claros com elementos escuros (ou vice-versa)
3. **Texto legível**: Evite fontes muito pequenas
4. **Formato vetorial convertido**: Se possível, exporte diagramas em alta resolução

## 🎨 Fontes de Diagramas de Exemplo

Se você precisa de exemplos para testar:

- **AWS Architecture Center**: https://aws.amazon.com/architecture/
- **Azure Architecture**: https://learn.microsoft.com/azure/architecture/
- **Google Cloud Architecture**: https://cloud.google.com/architecture
- **C4 Model Examples**: https://c4model.com/

## ⚠️ Importante

- Não commite imagens com informações sensíveis da sua empresa
- Remova dados confidenciais antes de adicionar imagens ao repositório
- Use o `.gitignore` para excluir imagens específicas se necessário

---

**Status**: Pronto para receber suas imagens de arquitetura! 🎯
