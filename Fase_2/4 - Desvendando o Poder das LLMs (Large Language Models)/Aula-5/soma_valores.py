import json

def somar_valores_apresentados(json_data):
    total = 0
    # Verifica se o JSON está no formato correto
    if not isinstance(json_data, dict):
        json_data = json.loads(json_data)
    
    # Navega até a lista de procedimentos
    procedimentos = json_data.get('value', {}).get('ListaProcedimento', [])
    
    # Para cada procedimento
    for procedimento in procedimentos:
        # Obtém os graus (onde estão os valores)
        graus = procedimento.get('ListaGraus', {})
        if graus:
            # Obtém o valor apresentado e multiplica pela quantidade
            valor = float(graus.get('ValorApresentado', 0))
            quantidade = int(graus.get('QtdApresentada', 1))
            total += valor * quantidade

    return total

# Lê o arquivo JSON
with open('dados.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Calcula o total
total = somar_valores_apresentados(json_data)
print(f"Valor total apresentado: R$ {total:,.2f}")

# Verificação com o ValorPago do JSON
valor_pago = float(json_data.get('value', {}).get('ValorPago', 0))
print(f"Valor pago registrado no JSON: R$ {valor_pago:,.2f}")

if abs(total - valor_pago) < 0.01:  # Compara com margem de erro para decimais
    print("Os valores conferem!")
else:
    print(f"Diferença entre valor calculado e valor pago: R$ {abs(total - valor_pago):,.2f}")
