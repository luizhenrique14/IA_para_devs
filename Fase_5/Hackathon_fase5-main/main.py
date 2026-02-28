"""
API REST para Análise de Ameaças STRIDE
FastAPI endpoint para análise de diagramas de arquitetura
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from analyzer import analyze_architecture
from pdf_generator import generate_stride_pdf_report
import tempfile
import os
from typing import Dict, Any

# Inicializar aplicação FastAPI
app = FastAPI(
    title="STRIDE Threat Analyzer API",
    description="API para análise de ameaças em diagramas de arquitetura usando metodologia STRIDE",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    Endpoint raiz - Informações da API
    """
    return {
        "message": "STRIDE Threat Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Verificação de saúde da API",
            "/analyze": "POST - Análise de diagrama de arquitetura (JSON)",
            "/analyze-pdf": "POST - Análise de diagrama de arquitetura (PDF)",
            "/stride-info": "Informações sobre metodologia STRIDE",
            "/docs": "Documentação interativa da API"
        }
    }


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    return {
        "status": "ok",
        "service": "STRIDE Threat Analyzer",
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analisa um diagrama de arquitetura e identifica ameaças usando STRIDE
    
    Args:
        image: Arquivo de imagem (PNG, JPG, JPEG) do diagrama de arquitetura
        
    Returns:
        JSON com a análise completa de ameaças STRIDE
    """
    
    # Validar tipo de arquivo
    allowed_types = ["image/png", "image/jpeg", "image/jpg"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado. Use PNG, JPG ou JPEG. Recebido: {image.content_type}"
        )
    
    # Validar tamanho do arquivo (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    
    try:
        # Salvar imagem temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail="Arquivo muito grande. Tamanho máximo: 10MB"
                )
            
            tmp.write(content)
            tmp_path = tmp.name
        
        # Analisar com LLM
        result = analyze_architecture(tmp_path)
        
        # Remover arquivo temporário
        os.unlink(tmp_path)
        
        # Verificar se houve erro na análise
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Erro na análise: {result['error']}"
            )
        
        return {
            "status": "success",
            "filename": image.filename,
            "analysis": result
        }
    
    except HTTPException:
        # Re-lançar exceções HTTP
        raise
    
    except Exception as e:
        # Limpar arquivo temporário em caso de erro
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar imagem: {str(e)}"
        )


@app.post("/analyze-pdf")
async def analyze_pdf(image: UploadFile = File(...)) -> FileResponse:
    """
    Analisa um diagrama de arquitetura e retorna relatório em PDF
    
    Args:
        image: Arquivo de imagem (PNG, JPG, JPEG) do diagrama de arquitetura
        
    Returns:
        Arquivo PDF com relatório completo de ameaças STRIDE
    """
    
    # Validar tipo de arquivo
    allowed_types = ["image/png", "image/jpeg", "image/jpg"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado. Use PNG, JPG ou JPEG. Recebido: {image.content_type}"
        )
    
    # Validar tamanho do arquivo (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    
    tmp_img_path = None
    tmp_pdf_path = None
    
    try:
        # Salvar imagem temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            content = await image.read()
            
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail="Arquivo muito grande. Tamanho máximo: 10MB"
                )
            
            tmp_img.write(content)
            tmp_img_path = tmp_img.name
        
        # Analisar com LLM
        result = analyze_architecture(tmp_img_path)
        
        # Verificar se houve erro na análise
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Erro na análise: {result['error']}"
            )
        
        # Gerar PDF
        tmp_pdf_path = tempfile.mktemp(suffix=".pdf")
        generate_stride_pdf_report(
            result,
            tmp_pdf_path,
            tmp_img_path
        )
        
        # Remover arquivo de imagem temporário
        os.unlink(tmp_img_path)
        tmp_img_path = None
        
        # Retornar PDF
        return FileResponse(
            path=tmp_pdf_path,
            media_type="application/pdf",
            filename=f"stride_report_{image.filename.split('.')[0]}.pdf",
            background=None  # Não deletar automaticamente, faremos isso depois
        )
    
    except HTTPException:
        # Re-lançar exceções HTTP
        raise
    
    except Exception as e:
        # Limpar arquivos temporários em caso de erro
        if tmp_img_path and os.path.exists(tmp_img_path):
            try:
                os.unlink(tmp_img_path)
            except:
                pass
        
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            try:
                os.unlink(tmp_pdf_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar PDF: {str(e)}"
        )


@app.get("/stride-info")
async def stride_info():
    """
    Retorna informações sobre a metodologia STRIDE
    """
    from stride_knowledge import STRIDE, STRIDE_DETAILS
    
    return {
        "methodology": "STRIDE",
        "description": "Metodologia de modelagem de ameaças desenvolvida pela Microsoft",
        "categories": STRIDE,
        "details": STRIDE_DETAILS
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando STRIDE Threat Analyzer API...")
    print("📖 Documentação: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
