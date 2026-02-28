"""
Script de Teste do Analisador STRIDE
Demonstra o uso básico do analyzer.py
"""

from analyzer import analyze_architecture, analyze_architecture_simple
import json
import os


def test_analyzer():
    """Teste básico do analisador"""
    
    print("🔍 Teste do Analisador STRIDE")
    print("="*80)
    
    # Verificar se a chave da API está configurada
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  AVISO: Variável OPENAI_API_KEY não encontrada")
        print("   Configure o arquivo .env com sua chave da OpenAI")
        print("   Copie .env.example para .env e adicione sua chave")
        return
    
    # Lista de imagens de exemplo para testar
    test_images = [
        "examples/arquitetura.png",
        "examples/diagrama.jpg",
        "examples/sistema.png"
    ]
    
    # Procurar por imagens disponíveis
    available_images = []
    for img in test_images:
        if os.path.exists(img):
            available_images.append(img)
    
    if not available_images:
        print("⚠️  Nenhuma imagem de exemplo encontrada na pasta 'examples/'")
        print("\n💡 Como testar:")
        print("   1. Adicione uma imagem de arquitetura na pasta 'examples/'")
        print("   2. Execute: python test_analyzer.py")
        print("   3. Ou use: python analyzer.py caminho/para/imagem.png")
        return
    
    # Testar com a primeira imagem disponível
    image_path = available_images[0]
    print(f"\n📸 Testando com: {image_path}")
    print("-"*80)
    
    # Realizar análise
    print("\n🔄 Enviando imagem para análise...")
    result = analyze_architecture(image_path)
    
    # Exibir resultado
    if result.get("success"):
        print("\n✅ Análise concluída com sucesso!")
        print("\n📊 Resultado:")
        print("="*80)
        print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
        print("="*80)
        
        # Exibir resumo se disponível
        if isinstance(result["analysis"], dict) and "resumo" in result["analysis"]:
            resumo = result["analysis"]["resumo"]
            print("\n📈 Resumo da Análise:")
            print(f"   • Total de Componentes: {resumo.get('total_componentes', 'N/A')}")
            print(f"   • Total de Ameaças: {resumo.get('total_ameacas', 'N/A')}")
            print(f"   • Ameaças Alta Criticidade: {resumo.get('ameacas_alta', 'N/A')}")
            print(f"   • Ameaças Média Criticidade: {resumo.get('ameacas_media', 'N/A')}")
            print(f"   • Ameaças Baixa Criticidade: {resumo.get('ameacas_baixa', 'N/A')}")
    else:
        print(f"\n❌ Erro na análise: {result.get('error')}")
    
    print("\n" + "="*80)
    print("✅ Teste concluído!")


def demo_stride_knowledge():
    """Demonstra o uso da base de conhecimento STRIDE"""
    from stride_knowledge import STRIDE, COMPONENT_THREATS, COUNTERMEASURES
    
    print("\n\n🛡️  Base de Conhecimento STRIDE")
    print("="*80)
    
    print("\n📚 Categorias STRIDE:")
    for categoria, descricao in STRIDE.items():
        print(f"   • {categoria}: {descricao}")
    
    print("\n🔍 Ameaças por Tipo de Componente:")
    for componente, ameacas in COMPONENT_THREATS.items():
        print(f"\n   {componente.upper()}:")
        for ameaca in ameacas:
            print(f"      - {ameaca}")
    
    print("\n🔒 Exemplo de Contramedidas (Spoofing):")
    for contramedida in COUNTERMEASURES["Spoofing"]:
        print(f"   • {contramedida}")


if __name__ == "__main__":
    # Executar teste do analisador
    test_analyzer()
    
    # Demonstrar base de conhecimento
    demo_stride_knowledge()
    
    print("\n\n💡 Próximos passos:")
    print("   1. Adicione mais imagens em 'examples/' para testar")
    print("   2. Implemente a Fase 4: API REST com FastAPI")
    print("   3. Implemente a Fase 5: Interface Web com Streamlit")
