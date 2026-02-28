"""
Interface Web STRIDE Threat Analyzer
Aplicação Streamlit para análise de ameaças em diagramas de arquitetura
"""

import streamlit as st
import requests
from PIL import Image
import io
import json
import tempfile
import os
from pdf_generator import generate_stride_pdf_report

# Configuração da página
st.set_page_config(
    page_title="STRIDE Threat Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar variáveis de sessão
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None
if 'pdf_data' not in st.session_state:
    st.session_state['pdf_data'] = None
if 'image_bytes' not in st.session_state:
    st.session_state['image_bytes'] = None
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = None

# URL da API (pode ser configurada)
API_URL = st.sidebar.text_input("URL da API", "http://localhost:8000")

# Título e descrição
st.title("🔒 STRIDE Threat Analyzer")
st.markdown("""
### Análise de Ameaças em Arquiteturas de Software

Esta ferramenta utiliza **Inteligência Artificial** e a metodologia **STRIDE** para identificar 
ameaças de segurança em diagramas de arquitetura de software.

**STRIDE** é uma metodologia desenvolvida pela Microsoft que categoriza ameaças em:
- **S**poofing (Falsificação de identidade)
- **T**ampering (Adulteração de dados)
- **R**epudiation (Repúdio de ações)
- **I**nformation Disclosure (Vazamento de informações)
- **D**enial of Service (Negação de serviço)
- **E**levation of Privilege (Elevação de privilégios)
""")

st.divider()

# Sidebar com informações
with st.sidebar:
    st.header("ℹ️ Informações")
    
    # Verificar status da API
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.ok:
            health_data = response.json()
            st.success("✅ API conectada")
            st.json(health_data)
        else:
            st.error("❌ API não está respondendo")
    except:
        st.error("❌ Não foi possível conectar à API")
        st.info("Certifique-se de que a API está rodando:\n```bash\nuvicorn main:app --reload\n```")
    
    st.divider()
    
    st.header("📚 Como usar")
    st.markdown("""
    1. Faça upload de um diagrama de arquitetura
    2. Clique em **Analisar Ameaças**
    3. Aguarde a análise da IA
    4. Revise as ameaças identificadas
    5. Implemente as contramedidas sugeridas
    """)
    
    st.divider()
    
    # Botão para ver metodologia STRIDE
    if st.button("📖 Ver Metodologia STRIDE"):
        try:
            response = requests.get(f"{API_URL}/stride-info")
            if response.ok:
                stride_data = response.json()
                st.json(stride_data)
        except:
            st.error("Erro ao buscar informações STRIDE")

# Área principal
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload do Diagrama")
    
    uploaded_file = st.file_uploader(
        "Selecione uma imagem do diagrama de arquitetura",
        type=["png", "jpg", "jpeg"],
        help="Formatos suportados: PNG, JPG, JPEG (máx. 10MB)"
    )
    
    if uploaded_file:
        # Mostrar preview da imagem
        image = Image.open(uploaded_file)
        st.image(image, caption="Diagrama carregado", use_container_width=True)
        
        # Informações do arquivo
        file_details = {
            "Nome": uploaded_file.name,
            "Tipo": uploaded_file.type,
            "Tamanho": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)

with col2:
    st.header("🔍 Análise de Ameaças")
    
    if uploaded_file:
        # Mostrar botão de análise apenas se não houver resultado
        if not st.session_state.get('analysis_result'):
            if st.button("🚀 Analisar Ameaças", type="primary", use_container_width=True):
                
                with st.spinner("🤖 Analisando arquitetura com IA... Isso pode levar alguns segundos..."):
                    try:
                        # Resetar o ponteiro do arquivo
                        uploaded_file.seek(0)
                        
                        # Fazer requisição para a API
                        files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        response = requests.post(
                            f"{API_URL}/analyze",
                            files=files,
                            timeout=60
                        )
                        
                        if response.ok:
                            result = response.json()
                            
                            st.success("✅ Análise concluída com sucesso!")
                            
                            # Armazenar resultado na sessão
                            st.session_state['analysis_result'] = result
                            st.session_state['file_name'] = uploaded_file.name
                            
                            # Extrair analysis corretamente (pode estar aninhado)
                            if 'analysis' in result:
                                analysis_data = result['analysis']
                                # Se analysis também tem 'analysis' dentro, pegar o conteúdo real
                                if isinstance(analysis_data, dict) and 'analysis' in analysis_data:
                                    st.session_state['analysis_data'] = analysis_data['analysis']
                                else:
                                    st.session_state['analysis_data'] = analysis_data
                            else:
                                st.session_state['analysis_data'] = result
                            
                            # Salvar conteúdo da imagem
                            uploaded_file.seek(0)
                            st.session_state['image_bytes'] = uploaded_file.read()
                            
                            # Forçar rerun para mostrar resultados
                            st.rerun()
                        else:
                            st.error(f"❌ Erro na análise: {response.status_code}")
                            st.json(response.json())
                            
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Timeout: A análise está demorando muito. Tente com uma imagem menor.")
                        
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Erro de conexão: Verifique se a API está rodando.")
                        st.code("uvicorn main:app --reload")
                        
                    except Exception as e:
                        st.error(f"❌ Erro inesperado: {str(e)}")
    
        # Mostrar resultado se existir na sessão
        if st.session_state.get('analysis_result'):
            result = st.session_state['analysis_result']
            
            st.subheader("📊 Resultado da Análise")
            
            # Se o resultado for um dict com 'analysis'
            if isinstance(result, dict) and "analysis" in result:
                analysis = result["analysis"]
                
                # Tentar parsear como JSON se for string
                if isinstance(analysis, str):
                    try:
                        analysis_json = json.loads(analysis)
                        st.json(analysis_json)
                    except:
                        st.markdown(analysis)
                else:
                    st.json(analysis)
            else:
                st.json(result)
            
            st.divider()
            
            # Opções de download lado a lado
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                st.download_button(
                    label="💾 Baixar Relatório (JSON)",
                    data=json.dumps(result, indent=2, ensure_ascii=False),
                    file_name=f"stride_analysis_{st.session_state.get('file_name', 'report')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_json"
                )
            
            with col_download2:
                # Botão para gerar PDF
                if not st.session_state.get('pdf_data'):
                    if st.button("📄 Gerar Relatório PDF", use_container_width=True, type="secondary", key="generate_pdf"):
                        with st.spinner("Gerando relatório PDF..."):
                            try:
                                # Salvar imagem temporariamente
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                                    tmp_img.write(st.session_state['image_bytes'])
                                    tmp_img_path = tmp_img.name
                                
                                # Gerar PDF
                                pdf_path = tempfile.mktemp(suffix=".pdf")
                                # Usar os dados de análise extraídos corretamente
                                analysis_data = st.session_state.get('analysis_data', result)
                                generate_stride_pdf_report(
                                    analysis_data,
                                    pdf_path,
                                    tmp_img_path
                                )
                                
                                # Ler PDF gerado
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_data = pdf_file.read()
                                
                                # Limpar arquivos temporários
                                os.unlink(tmp_img_path)
                                os.unlink(pdf_path)
                                
                                # Armazenar PDF na sessão
                                st.session_state['pdf_data'] = pdf_data
                                st.success("✅ PDF gerado com sucesso!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Erro ao gerar PDF: {str(e)}")
                else:
                    # Mostrar botão de download do PDF se já foi gerado
                    st.download_button(
                        label="⬇️ Baixar PDF Gerado",
                        data=st.session_state['pdf_data'],
                        file_name=f"stride_report_{st.session_state.get('file_name', 'report').split('.')[0]}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf"
                    )
            
            st.divider()
            
            # Botão para nova análise
            if st.button("🔄 Nova Análise", type="primary", use_container_width=True):
                # Limpar sessão
                st.session_state['analysis_result'] = None
                st.session_state['analysis_data'] = None
                st.session_state['pdf_data'] = None
                st.session_state['image_bytes'] = None
                st.session_state['file_name'] = None
                st.rerun()
    else:
        st.info("👆 Faça upload de um diagrama para começar a análise")

# Rodapé
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🎓 Projeto Hackathon FIAP - Fase 5</p>
    <p>Desenvolvido com ❤️ usando Streamlit, FastAPI e OpenAI GPT-4 Vision</p>
</div>
""", unsafe_allow_html=True)

# Exemplos na sidebar
with st.sidebar:
    st.divider()
    st.header("💡 Dicas")
    st.markdown("""
    **Para melhores resultados:**
    - Use diagramas claros e legíveis
    - Inclua labels nos componentes
    - Mostre conexões entre sistemas
    - Identifique trust boundaries
    - Indique fluxo de dados
    """)
