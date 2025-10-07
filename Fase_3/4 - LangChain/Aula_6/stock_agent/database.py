"""
Database connection and configuration module.
This module handles all database-related operations and connections.
"""

import os
from dotenv import load_dotenv
import psycopg2
from typing import Optional

# Load environment variables
load_dotenv()

def get_db_config() -> dict:
    """
    Get database configuration from environment variables.
    """
    return {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': 'localhost',
        'port': os.getenv('DB_PORT')
    }

def conectar_db() -> Optional[psycopg2.extensions.connection]:
    """
    Create a connection to the PostgreSQL database.
    
    Returns:
        psycopg2.extensions.connection: Database connection object if successful
        None: If connection fails
    """
    try:
        config = get_db_config()
        conn = psycopg2.connect(**config)
        print("Conexão ao banco de dados PostgreSQL bem-sucedida!")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        raise

def execute_query(query: str, params: tuple = None) -> list:
    """
    Execute a database query and return the results.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Query parameters. Defaults to None.
    
    Returns:
        list: Query results
    """
    conn = conectar_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()