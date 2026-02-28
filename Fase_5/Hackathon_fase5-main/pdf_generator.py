"""
Gerador de Relatórios PDF para Análise STRIDE
Cria relatórios profissionais de modelagem de ameaças
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.platypus.flowables import HRFlowable
from datetime import datetime
import json
import os


class STRIDEReportGenerator:
    """Gerador de relatórios PDF para análises STRIDE"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Cria estilos personalizados para o relatório"""
        
        # Título principal
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtítulo
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Seção
        self.styles.add(ParagraphStyle(
            name='CustomSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#0066cc'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Categoria STRIDE
        self.styles.add(ParagraphStyle(
            name='STRIDECategory',
            parent=self.styles['Heading4'],
            fontSize=12,
            textColor=colors.HexColor('#cc0000'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Texto normal
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))
        
        # Texto de destaque
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#0066cc'),
            fontName='Helvetica-Bold'
        ))
    
    def generate_report(self, analysis_data: dict, output_path: str, image_path: str = None):
        """
        Gera o relatório PDF completo
        
        Args:
            analysis_data: Dados da análise (pode ser dict ou string JSON)
            output_path: Caminho para salvar o PDF
            image_path: Caminho da imagem do diagrama (opcional)
        """
        
        # Parsear dados se for string
        if isinstance(analysis_data, str):
            try:
                analysis_data = json.loads(analysis_data)
            except json.JSONDecodeError:
                # Se não for JSON válido, mantém como um dict com o texto
                analysis_data = {"analysis_text": analysis_data}
        
        # Criar documento
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container para elementos do documento
        story = []
        
        # Adicionar capa
        story.extend(self._create_cover_page(analysis_data))
        story.append(PageBreak())
        
        # Adicionar sumário executivo
        story.extend(self._create_executive_summary(analysis_data))
        story.append(Spacer(1, 0.3*inch))
        
        # Adicionar imagem do diagrama se disponível
        if image_path and os.path.exists(image_path):
            story.extend(self._add_diagram_image(image_path))
            story.append(PageBreak())
        
        # Adicionar análise detalhada
        story.extend(self._create_detailed_analysis(analysis_data))
        
        # Adicionar metodologia STRIDE
        story.append(PageBreak())
        story.extend(self._create_stride_methodology())
        
        # Gerar PDF
        doc.build(story)
        
        return output_path
    
    def _create_cover_page(self, data: dict):
        """Cria a página de capa do relatório"""
        elements = []
        
        # Espaço inicial
        elements.append(Spacer(1, 2*inch))
        
        # Título
        elements.append(Paragraph(
            "🔒 Relatório de Modelagem de Ameaças",
            self.styles['CustomTitle']
        ))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Subtítulo
        elements.append(Paragraph(
            "Análise STRIDE de Segurança em Arquitetura de Software",
            self.styles['CustomSubtitle']
        ))
        
        elements.append(Spacer(1, 1*inch))
        
        # Informações do relatório
        info_data = [
            ['Data de Geração:', datetime.now().strftime("%d/%m/%Y às %H:%M")],
            ['Metodologia:', 'STRIDE (Microsoft)'],
            ['Ferramenta:', 'STRIDE Threat Analyzer - AI Powered'],
            ['Status:', 'Análise Concluída']
        ]
        
        info_table = Table(info_data, colWidths=[2.5*inch, 3.5*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0066cc')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(info_table)
        
        elements.append(Spacer(1, 1*inch))
        
        # Rodapé da capa
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0066cc')))
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(
            "<i>Projeto Hackathon FIAP - Fase 5<br/>Desenvolvido com IA e OpenAI GPT-4 Vision</i>",
            self.styles['Normal']
        ))
        
        return elements
    
    def _create_executive_summary(self, data: dict):
        """Cria o sumário executivo"""
        elements = []
        
        elements.append(Paragraph("Sumário Executivo", self.styles['CustomTitle']))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0066cc')))
        elements.append(Spacer(1, 0.2*inch))
        
        # Extrair informações resumidas
        total_componentes = 0
        total_ameacas = 0
        ameacas_criticas = 0
        
        if isinstance(data, dict):
            # Tentar extrair do resumo se existir
            if 'resumo' in data:
                resumo = data['resumo']
                total_componentes = resumo.get('total_componentes', 0)
                total_ameacas = resumo.get('total_ameacas', 0)
                ameacas_criticas = resumo.get('ameacas_alta', 0)
            # Ou contar dos componentes
            elif 'componentes' in data:
                total_componentes = len(data['componentes'])
                for comp in data['componentes']:
                    if 'ameacas' in comp:
                        total_ameacas += len(comp['ameacas'])
        
        # Tabela de métricas
        metrics_data = [
            ['📊 Métrica', '📈 Valor', '🎯 Status'],
            ['Componentes Analisados', str(total_componentes), 'Completo'],
            ['Ameaças Identificadas', str(total_ameacas), 'Alta Prioridade' if total_ameacas > 10 else 'Atenção'],
            ['Ameaças Críticas', str(ameacas_criticas), 'Crítico' if ameacas_criticas > 0 else 'OK']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Descrição do sumário
        elements.append(Paragraph(
            "<b>Visão Geral da Análise</b>",
            self.styles['CustomSection']
        ))
        
        summary_text = """
        Este relatório apresenta uma análise completa de segurança baseada na metodologia STRIDE 
        (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege).
        A análise foi realizada utilizando Inteligência Artificial para identificar potenciais ameaças 
        de segurança na arquitetura do sistema e fornecer recomendações práticas para mitigação.
        """
        
        elements.append(Paragraph(summary_text, self.styles['CustomBody']))
        
        return elements
    
    def _add_diagram_image(self, image_path: str):
        """Adiciona a imagem do diagrama ao relatório"""
        elements = []
        
        elements.append(Paragraph("Diagrama de Arquitetura Analisado", self.styles['CustomSection']))
        elements.append(Spacer(1, 0.15*inch))
        
        try:
            # Adicionar imagem com tamanho ajustado
            img = Image(image_path, width=6*inch, height=4*inch, kind='proportional')
            elements.append(img)
        except:
            elements.append(Paragraph(
                "<i>Imagem do diagrama não disponível</i>",
                self.styles['Normal']
            ))
        
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_detailed_analysis(self, data: dict):
        """Cria a análise detalhada das ameaças"""
        elements = []
        
        elements.append(Paragraph("Análise Detalhada de Ameaças", self.styles['CustomTitle']))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0066cc')))
        elements.append(Spacer(1, 0.2*inch))
        
        # Extrair dados aninhados se necessário
        actual_data = data
        if isinstance(data, dict):
            # Se tiver 'analysis' aninhado, extrair
            if 'analysis' in data and isinstance(data['analysis'], dict):
                actual_data = data['analysis']
            # Se tiver 'success' e 'analysis', é resposta do analyzer
            elif 'success' in data and 'analysis' in data:
                actual_data = data['analysis']
        
        # Se tiver estrutura de componentes
        if isinstance(actual_data, dict) and 'componentes' in actual_data:
            for idx, componente in enumerate(actual_data['componentes'], 1):
                elements.extend(self._format_component_analysis(componente, idx))
            
            # Adicionar fluxos de dados se existir
            if 'fluxos_dados' in actual_data and actual_data['fluxos_dados']:
                elements.append(Spacer(1, 0.3*inch))
                elements.append(Paragraph("Fluxos de Dados e Comunicação", self.styles['CustomSection']))
                elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
                elements.append(Spacer(1, 0.1*inch))
                
                for idx, fluxo in enumerate(actual_data['fluxos_dados'], 1):
                    elements.append(Paragraph(
                        f"<b>{idx}. {fluxo.get('origem', 'N/A')} → {fluxo.get('destino', 'N/A')}</b>",
                        self.styles['CustomBody']
                    ))
                    if 'ameacas' in fluxo:
                        elements.append(Paragraph("Ameaças:", self.styles['Highlight']))
                        for ameaca in fluxo['ameacas']:
                            elements.append(Paragraph(f"• {ameaca}", self.styles['CustomBody']))
                    if 'contramedidas' in fluxo:
                        elements.append(Paragraph("Contramedidas:", self.styles['Highlight']))
                        for contra in fluxo['contramedidas']:
                            elements.append(Paragraph(f"✓ {contra}", self.styles['CustomBody']))
                    elements.append(Spacer(1, 0.15*inch))
            
            # Adicionar recomendações gerais
            if 'recomendacoes_gerais' in actual_data and actual_data['recomendacoes_gerais']:
                elements.append(Spacer(1, 0.3*inch))
                elements.append(Paragraph("Recomendações Gerais de Segurança", self.styles['CustomSection']))
                elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
                elements.append(Spacer(1, 0.1*inch))
                for rec in actual_data['recomendacoes_gerais']:
                    elements.append(Paragraph(f"✓ {rec}", self.styles['CustomBody']))
        
        # Se tiver texto de análise (string)
        elif isinstance(actual_data, dict) and 'analysis_text' in actual_data:
            text = actual_data['analysis_text'].replace('\n', '<br/>')
            elements.append(Paragraph(text, self.styles['CustomBody']))
        
        # Se for string diretamente
        elif isinstance(actual_data, str):
            # Tentar parsear como JSON
            try:
                analysis_json = json.loads(actual_data)
                elements.extend(self._format_json_analysis(analysis_json))
            except:
                # Se não for JSON, adicionar como texto formatado
                elements.append(Paragraph(actual_data.replace('\n', '<br/>'), self.styles['CustomBody']))
        
        # Fallback: mostrar estrutura JSON
        else:
            elements.append(Paragraph(
                "<i>Formato de análise não reconhecido. Dados brutos:</i>",
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 0.1*inch))
            json_str = json.dumps(actual_data, indent=2, ensure_ascii=False)
            # Dividir em linhas para melhor formatação
            for line in json_str.split('\n')[:50]:  # Limitar a 50 linhas
                elements.append(Paragraph(line, self.styles['Code']))
        
        return elements
    
    def _format_component_analysis(self, component: dict, number: int):
        """Formata a análise de um componente"""
        elements = []
        
        # Nome do componente
        comp_name = component.get('nome', f'Componente {number}')
        elements.append(Paragraph(
            f"{number}. {comp_name}",
            self.styles['CustomSection']
        ))
        
        # Tipo
        if 'tipo' in component:
            elements.append(Paragraph(
                f"<b>Tipo:</b> {component['tipo']}",
                self.styles['CustomBody']
            ))
        
        # Descrição
        if 'descricao' in component:
            elements.append(Paragraph(
                f"<b>Descrição:</b> {component['descricao']}",
                self.styles['CustomBody']
            ))
        
        elements.append(Spacer(1, 0.1*inch))
        
        # Ameaças identificadas
        if 'ameacas' in component:
            elements.append(Paragraph("<b>Ameaças Identificadas:</b>", self.styles['Highlight']))
            
            for ameaca in component['ameacas']:
                categoria = ameaca.get('categoria', 'N/A')
                descricao = ameaca.get('descricao', 'N/A')
                severidade = ameaca.get('severidade', 'Média')
                
                elements.append(Paragraph(
                    f"• <b>[{categoria}]</b> - {severidade}: {descricao}",
                    self.styles['CustomBody']
                ))
        
        # Contramedidas
        if 'contramedidas' in component:
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph("<b>Contramedidas Recomendadas:</b>", self.styles['Highlight']))
            
            for contramedida in component['contramedidas']:
                elements.append(Paragraph(
                    f"✓ {contramedida}",
                    self.styles['CustomBody']
                ))
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _format_json_analysis(self, data: dict):
        """Formata análise que está em formato JSON"""
        elements = []
        
        # Iterar pelas chaves principais
        for key, value in data.items():
            if key in ['componentes', 'components']:
                if isinstance(value, list):
                    for idx, comp in enumerate(value, 1):
                        elements.extend(self._format_component_analysis(comp, idx))
            else:
                # Adicionar outras seções
                elements.append(Paragraph(
                    key.replace('_', ' ').title(),
                    self.styles['CustomSection']
                ))
                
                if isinstance(value, (list, dict)):
                    elements.append(Paragraph(
                        "<pre>" + json.dumps(value, indent=2, ensure_ascii=False) + "</pre>",
                        self.styles['Code']
                    ))
                else:
                    elements.append(Paragraph(str(value), self.styles['CustomBody']))
                
                elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_stride_methodology(self):
        """Cria a seção sobre metodologia STRIDE"""
        elements = []
        
        from stride_knowledge import STRIDE, STRIDE_DETAILS
        
        elements.append(Paragraph("Sobre a Metodologia STRIDE", self.styles['CustomTitle']))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0066cc')))
        elements.append(Spacer(1, 0.2*inch))
        
        intro_text = """
        STRIDE é uma metodologia de modelagem de ameaças desenvolvida pela Microsoft que fornece 
        um framework sistemático para identificar ameaças à segurança em sistemas de software. 
        O acrônimo representa seis categorias de ameaças:
        """
        
        elements.append(Paragraph(intro_text, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Tabela com categorias STRIDE
        stride_data = [['Categoria', 'Descrição', 'Foco']]
        
        stride_info = {
            'S - Spoofing': ('Falsificação de identidade', 'Autenticação'),
            'T - Tampering': ('Adulteração de dados', 'Integridade'),
            'R - Repudiation': ('Negação de ações', 'Não-repúdio'),
            'I - Information Disclosure': ('Vazamento de dados', 'Confidencialidade'),
            'D - Denial of Service': ('Negação de serviço', 'Disponibilidade'),
            'E - Elevation of Privilege': ('Elevação de privilégios', 'Autorização')
        }
        
        for cat, (desc, foco) in stride_info.items():
            stride_data.append([cat, desc, foco])
        
        stride_table = Table(stride_data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
        stride_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(stride_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Nota final
        footer_text = """
        <b>Nota:</b> Este relatório foi gerado automaticamente usando Inteligência Artificial 
        (GPT-4 Vision) para análise de diagramas de arquitetura. As ameaças e recomendações 
        identificadas devem ser revisadas por especialistas em segurança para validação e 
        adaptação ao contexto específico do projeto.
        """
        
        elements.append(Paragraph(footer_text, self.styles['CustomBody']))
        
        return elements


def generate_stride_pdf_report(analysis_data, output_path: str, image_path: str = None):
    """
    Função helper para gerar relatório PDF
    
    Args:
        analysis_data: Dados da análise STRIDE
        output_path: Caminho do arquivo PDF a ser gerado
        image_path: Caminho da imagem do diagrama (opcional)
    
    Returns:
        str: Caminho do arquivo PDF gerado
    """
    generator = STRIDEReportGenerator()
    return generator.generate_report(analysis_data, output_path, image_path)
