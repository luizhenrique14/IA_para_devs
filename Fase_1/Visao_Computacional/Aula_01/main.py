import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('face.png')

if imagem is None:
    print("Erro: Imagem não encontrada. Verifique o caminho e o nome do arquivo.")
    exit()

imagem_suavizada = cv2.GaussianBlur(imagem, (15, 15), 0)

imagem_rgb = cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2RGB)

plt.imshow(imagem_rgb)

plt.show()

# conversao de imagem para grayscale

bordas = cv2.Canny(imagem_suavizada, 100, 200)

plt.imshow(bordas, cmap='gray')
plt.axis('off')  # Remove os eixos para melhor visualização
plt.show()