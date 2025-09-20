import json
from typing import Dict, Any

def calcular_total_procedimento(procedimento: Dict[str, Any]) -> float:
    """Calcula o total para um procedimento específico."""
    graus = procedimento.get('ListaGraus', {})
    if not graus:
        return 0.0
    
    try:
        valor = float(graus.get('ValorApresentado', 0))
        quantidade = int(graus.get('QtdApresentada', 1))
        return valor * quantidade
    except (ValueError, TypeError):
        print(f"Erro ao processar valores para o procedimento: {procedimento.get('Procedimento', 'Desconhecido')}")
        return 0.0

def analisar_procedimentos(json_data: Dict[str, Any]) -> None:
    """Analisa e imprime os detalhes dos procedimentos e seus valores."""
    if not isinstance(json_data, dict):
        json_data = json.loads(json_data)
    
    valor_pago = float(json_data.get('value', {}).get('ValorPago', 0))
    procedimentos = json_data.get('value', {}).get('ListaProcedimento', [])
    
    total = 0.0
    print("\nDetalhamento dos procedimentos:")
    print("-" * 80)
    
    # Dicionário para agrupar valores por tipo de profissional
    valores_por_profissional = {}
    
    for procedimento in procedimentos:
        nome_proc = procedimento.get('Procedimento', 'Desconhecido')
        graus = procedimento.get('ListaGraus', {})
        
        if graus:
            valor = float(graus.get('ValorApresentado', 0))
            qtd = int(graus.get('QtdApresentada', 1))
            subtotal = valor * qtd
            profissional = graus.get('Descricao', 'Não especificado')
            
            # Agrupa valores por profissional
            if profissional not in valores_por_profissional:
                valores_por_profissional[profissional] = 0
            valores_por_profissional[profissional] += subtotal
            
            print(f"Procedimento: {nome_proc}")
            print(f"Profissional: {profissional}")
            print(f"Valor: R$ {valor:,.2f} x {qtd} = R$ {subtotal:,.2f}")
            print("-" * 80)
            
            total += subtotal
    
    print("\nResumo por profissional:")
    print("-" * 80)
    for profissional, valor in valores_por_profissional.items():
        print(f"{profissional}: R$ {valor:,.2f}")
    
    print("\nResumo geral:")
    print("-" * 80)
    print(f"Total calculado: R$ {total:,.2f}")
    print(f"Valor pago registrado: R$ {valor_pago:,.2f}")
    
    diferenca = abs(total - valor_pago)
    if diferenca < 0.01:
        print("\nOs valores conferem!")
    else:
        print(f"\nDiferença encontrada: R$ {diferenca:,.2f}")

# Processa o JSON fornecido
json_str = """SEU_JSON_AQUI"""
try:
    json_data = json.loads(json_str)
    analisar_procedimentos(json_data)
except json.JSONDecodeError as e:
    print(f"Erro ao decodificar o JSON: {e}")
except Exception as e:
    print(f"Erro inesperado: {e}")
