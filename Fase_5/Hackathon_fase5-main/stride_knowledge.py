"""
Base de Conhecimento STRIDE - Metodologia de Modelagem de Ameaças
"""

# Definição da metodologia STRIDE
STRIDE = {
    "S - Spoofing": "Falsificação de identidade",
    "T - Tampering": "Adulteração de dados",
    "R - Repudiation": "Negação de ações realizadas",
    "I - Information Disclosure": "Vazamento de informações",
    "D - Denial of Service": "Negação de serviço",
    "E - Elevation of Privilege": "Elevação de privilégios"
}

# Ameaças por componente
COMPONENT_THREATS = {
    "database": ["SQL Injection", "Data Leakage", "Unauthorized Access"],
    "api": ["Broken Authentication", "Injection Attacks", "Rate Limiting"],
    "server": ["DDoS", "Privilege Escalation", "Malware"],
    "user": ["Phishing", "Session Hijacking", "Credential Theft"],
    "cloud": ["Misconfiguration", "Data Breach", "Account Hijacking"]
}

# Descrições detalhadas das categorias STRIDE
STRIDE_DETAILS = {
    "Spoofing": {
        "description": "Ameaças relacionadas à falsificação de identidade ou credenciais",
        "examples": [
            "Falsificação de autenticação",
            "Roubo de credenciais",
            "Impersonação de usuário",
            "Falsificação de tokens"
        ]
    },
    "Tampering": {
        "description": "Ameaças relacionadas à modificação maliciosa de dados",
        "examples": [
            "Modificação de dados em trânsito",
            "Alteração de logs",
            "Modificação de código",
            "Corrupção de banco de dados"
        ]
    },
    "Repudiation": {
        "description": "Ameaças relacionadas à negação de ações realizadas",
        "examples": [
            "Falta de auditoria",
            "Logs insuficientes",
            "Ausência de assinatura digital",
            "Não rastreabilidade de ações"
        ]
    },
    "Information Disclosure": {
        "description": "Ameaças relacionadas ao vazamento ou exposição de informações sensíveis",
        "examples": [
            "Exposição de dados sensíveis",
            "Vazamento de credenciais",
            "Informações em logs",
            "Mensagens de erro verbosas"
        ]
    },
    "Denial of Service": {
        "description": "Ameaças que visam tornar o sistema indisponível",
        "examples": [
            "Ataques DDoS",
            "Esgotamento de recursos",
            "Flooding de requisições",
            "Crash de aplicação"
        ]
    },
    "Elevation of Privilege": {
        "description": "Ameaças relacionadas à obtenção de privilégios não autorizados",
        "examples": [
            "Exploração de vulnerabilidades",
            "Bypass de controles de acesso",
            "Escalação de privilégios",
            "Execução de código arbitrário"
        ]
    }
}

# Contramedidas comuns por categoria STRIDE
COUNTERMEASURES = {
    "Spoofing": [
        "Implementar autenticação forte (MFA)",
        "Usar certificados digitais",
        "Validar tokens e sessões",
        "Implementar OAuth/OpenID Connect"
    ],
    "Tampering": [
        "Usar HTTPS/TLS para comunicação",
        "Implementar assinaturas digitais",
        "Validar integridade de dados",
        "Proteger logs contra modificação"
    ],
    "Repudiation": [
        "Implementar logging abrangente",
        "Usar assinaturas digitais",
        "Implementar auditoria de ações",
        "Garantir não repúdio em transações críticas"
    ],
    "Information Disclosure": [
        "Criptografar dados sensíveis",
        "Implementar controles de acesso rigorosos",
        "Sanitizar mensagens de erro",
        "Usar HTTPS para todas as comunicações"
    ],
    "Denial of Service": [
        "Implementar rate limiting",
        "Usar CDN e cache",
        "Configurar timeouts adequados",
        "Implementar circuit breakers"
    ],
    "Elevation of Privilege": [
        "Implementar princípio do menor privilégio",
        "Validar todas as entradas",
        "Usar controles de acesso baseados em função (RBAC)",
        "Manter sistema atualizado"
    ]
}

def get_stride_info():
    """Retorna informações completas sobre a metodologia STRIDE"""
    return {
        "methodology": STRIDE,
        "details": STRIDE_DETAILS,
        "countermeasures": COUNTERMEASURES,
        "component_threats": COMPONENT_THREATS
    }
