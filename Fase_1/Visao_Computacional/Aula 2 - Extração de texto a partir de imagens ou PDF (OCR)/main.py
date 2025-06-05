# pip install pytesseract opencv-python pymupdf


# 📌 Além disso, você precisa ter o Tesseract OCR instalado no seu sistema:

# Windows: https://github.com/tesseract-ocr/tesseract/wiki

# Após instalar, adicione o caminho do executável ao script, como mostrado abaixo.

import cv2
import pytesseract
import fitz  # PyMuPDF
import os

# 🧠 (Somente para Windows) Defina o caminho para o executável do Tesseract:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------- Função para extrair texto de imagem ----------
def ocr_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
    text = pytesseract.image_to_string(gray, lang='por')  # 'eng' para inglês, 'por' para português
    return text

# ---------- Função para extrair texto de PDF ----------
def ocr_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()  # Converte a página em imagem
        image_path = f"page_{page_num}.png"
        pix.save(image_path)  # Salva a imagem temporariamente

        # OCR da imagem da página
        text += f"\n--- Página {page_num + 1} ---\n"
        text += ocr_from_image(image_path)

        os.remove(image_path)  # Remove imagem temporária
    return text

# ---------- Execução principal ----------
if __name__ == "__main__":
    print("Escolha o tipo de entrada:")
    print("1 - Imagem (.png, .jpg)")
    print("2 - PDF (.pdf)")
    escolha = input("Digite 1 ou 2: ")

    if escolha == "1":
        caminho = input("Digite o caminho da imagem: ")
        resultado = ocr_from_image(caminho)
        print("\n📝 Texto extraído da imagem:\n")
        print(resultado)

    elif escolha == "2":
        caminho = input("Digite o caminho do PDF: ")
        resultado = ocr_from_pdf(caminho)
        print("\n📝 Texto extraído do PDF:\n")
        print(resultado)

    else:
        print("Opção inválida.")
