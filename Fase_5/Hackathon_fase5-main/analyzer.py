"""
Analisador de Arquitetura com GPT-4 Vision
Aplica a metodologia STRIDE em diagramas de arquitetura
"""

import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import json

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_architecture(image_path: str) -> dict:
    """
    Analisa um diagrama de arquitetura usando GPT-4 Vision e aplica STRIDE
    
    Args:
        image_path: Caminho para o arquivo de imagem da arquitetura
        
    Returns:
        dict: Resultado da análise com ameaças identificadas e contramedidas
    """
    
    # Ler e codificar a imagem
    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return {"error": f"Arquivo não encontrado: {image_path}"}
    except Exception as e:
        return {"error": f"Erro ao ler arquivo: {str(e)}"}
    
    # Prompt para análise STRIDE
    prompt = """
    Analise este diagrama de arquitetura de software e aplique a metodologia STRIDE de modelagem de ameaças.
    
    Siga estas instruções:
    
    1. **Identificação de Componentes**: Liste todos os componentes visíveis no diagrama (usuários, servidores, bancos de dados, APIs, serviços externos, redes, etc)
    
    2. **Análise STRIDE por Componente**: Para cada componente identificado, analise as seguintes categorias:
       - **S (Spoofing)**: Ameaças de falsificação de identidade
       - **T (Tampering)**: Ameaças de adulteração de dados
       - **R (Repudiation)**: Ameaças de repúdio/negação de ações
       - **I (Information Disclosure)**: Ameaças de vazamento de informações
       - **D (Denial of Service)**: Ameaças de negação de serviço
       - **E (Elevation of Privilege)**: Ameaças de elevação de privilégio
    
    3. **Análise de Fluxos de Dados**: Identifique os fluxos de dados entre componentes e possíveis ameaças nessas comunicações
    
    4. **Contramedidas**: Para cada ameaça identificada, sugira contramedidas específicas e práticas
    
    5. **Priorização**: Classifique as ameaças por criticidade (Alta, Média, Baixa)
    
    Estruture a resposta em formato JSON com a seguinte estrutura:
    {
        "componentes": [
            {
                "nome": "nome do componente",
                "tipo": "tipo do componente (ex: database, api, user, server)",
                "ameacas": [
                    {
                        "categoria_stride": "S/T/R/I/D/E",
                        "descricao": "descrição da ameaça",
                        "criticidade": "Alta/Média/Baixa",
                        "contramedidas": ["contramedida 1", "contramedida 2"]
                    }
                ]
            }
        ],
        "fluxos_dados": [
            {
                "origem": "componente origem",
                "destino": "componente destino",
                "ameacas": ["ameaça 1", "ameaça 2"],
                "contramedidas": ["contramedida 1", "contramedida 2"]
            }
        ],
        "resumo": {
            "total_componentes": 0,
            "total_ameacas": 0,
            "ameacas_alta": 0,
            "ameacas_media": 0,
            "ameacas_baixa": 0
        },
        "recomendacoes_gerais": ["recomendação 1", "recomendação 2"]
    }
    
    Seja específico e detalhado nas análises. Considere as melhores práticas de segurança atuais.
    """
    
    try:
        # Fazer chamada à API do OpenAI com GPT-4 Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.7
        )
        
        # Extrair resposta
        analysis_text = response.choices[0].message.content
        
        # Tentar parsear como JSON
        try:
            # Procurar por JSON na resposta
            if "```json" in analysis_text:
                json_start = analysis_text.find("```json") + 7
                json_end = analysis_text.find("```", json_start)
                json_str = analysis_text[json_start:json_end].strip()
                analysis_json = json.loads(json_str)
            elif analysis_text.strip().startswith("{"):
                analysis_json = json.loads(analysis_text)
            else:
                # Se não for JSON, retornar como texto
                analysis_json = {
                    "analysis_text": analysis_text,
                    "note": "Análise retornada em formato texto"
                }
        except json.JSONDecodeError:
            # Se falhar o parse, retornar como texto
            analysis_json = {
                "analysis_text": analysis_text,
                "note": "Não foi possível parsear como JSON"
            }
        
        return {
            "success": True,
            "analysis": analysis_json,
            "model": "gpt-4o",
            "image_path": image_path
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Erro na análise: {str(e)}"
        }


def analyze_architecture_simple(image_path: str) -> str:
    """
    Versão simplificada que retorna apenas o texto da análise
    
    Args:
        image_path: Caminho para o arquivo de imagem
        
    Returns:
        str: Texto da análise
    """
    result = analyze_architecture(image_path)
    
    if not result.get("success", False):
        return f"Erro: {result.get('error', 'Erro desconhecido')}"
    
    analysis = result.get("analysis", {})
    
    if "analysis_text" in analysis:
        return analysis["analysis_text"]
    
    # Se for JSON, formatar de forma legível
    return json.dumps(analysis, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Teste do analisador
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python analyzer.py <caminho_imagem>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"Analisando: {image_path}")
    print("-" * 80)
    
    result = analyze_architecture(image_path)
    
    if result.get("success"):
        print("✅ Análise concluída com sucesso!")
        print("\nResultado:")
        print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
    else:
        print(f"❌ Erro: {result.get('error')}")
