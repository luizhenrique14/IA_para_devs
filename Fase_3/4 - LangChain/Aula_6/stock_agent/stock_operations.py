"""
Stock operations module.
This module handles all stock-related database operations and data processing.
"""

from typing import List, Tuple, Optional
from database import execute_query

def obter_dados_acoes(termo_busca: str) -> List[Tuple]:
    """
    Retrieve stock data based on a search term.
    
    Args:
        termo_busca (str): Search term for stock name or ticker
    
    Returns:
        List[Tuple]: List of stock data records
    """
    query = """
    SELECT s.ticker, s.name, sp.date, sp.open, sp.high, sp.low, sp.close, sp.volume
    FROM stocks s
    JOIN stock_prices sp ON s.id = sp.stock_id
    WHERE s.name ILIKE %s OR s.ticker ILIKE %s;
    """
    params = (f"%{termo_busca}%", f"%{termo_busca}%")
    return execute_query(query, params)

def formatar_dados_acoes(dados: List[Tuple]) -> str:
    """
    Format stock data into a readable string.
    
    Args:
        dados (List[Tuple]): List of stock data records
    
    Returns:
        str: Formatted string with stock information
    """
    if not dados:
        return "Nenhum dado encontrado."
    
    resultado = ""
    for dado in dados:
        resultado += (
            f"Ticker: {dado[0]}, Nome: {dado[1]}, Data: {dado[2]}, "
            f"Abertura: {dado[3]}, Máxima: {dado[4]}, Mínima: {dado[5]}, "
            f"Fechamento: {dado[6]}, Volume: {dado[7]}\n"
        )
    return resultado