# Se necessário, instale as bibliotecas antes de rodar o código:
# pip install numpy matplotlib scikit-learn pillow

# Importa a biblioteca PIL para manipulação de imagens (não usada diretamente aqui, mas útil para outros formatos)
from PIL import Image
# Importa glob para buscar arquivos em diretórios (não usado diretamente neste exemplo)
import glob
# Importa numpy para manipulação de arrays numéricos (essencial para processar imagens)
import numpy as np

# Importa matplotlib para visualização de imagens e gráficos
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

# Importa o KMeans do scikit-learn para realizar a clusterização (agrupamento)
from sklearn.cluster import KMeans

# Carrega as imagens de mamografia do disco usando o matplotlib.image
img_G = mpimg.imread('mdb001.pgm') # Carrega a imagem do tipo G
img_D = mpimg.imread('mdb003.pgm') # Carrega a imagem do tipo D
img_F = mpimg.imread('mdb005.pgm') # Carrega a imagem do tipo F

# Plota as imagens originais para visualização
fig, axs = plt.subplots(1, 3, figsize=(10, 3)) # Cria uma figura com 3 subplots lado a lado
im1 = axs[0].imshow(img_G, cmap='gray', vmin=0, vmax=255) # Mostra a imagem G em escala de cinza
axs[0].set_title('Original G') # Título do subplot
im2 = axs[1].imshow(img_D, cmap='gray', vmin=0, vmax=255) # Mostra a imagem D em escala de cinza
axs[1].set_title('Original D') # Título do subplot
im3 = axs[2].imshow(img_F, cmap='gray', vmin=0, vmax=255) # Mostra a imagem F em escala de cinza
axs[2].set_title('Original F') # Título do subplot
plt.show() # Exibe as imagens

# Define uma função para segmentar a imagem usando KMeans
def filtro_kmeans(img, clusters):
    # Redimensiona a imagem para um vetor de pixels (necessário para o KMeans)
    vectorized = img.reshape((-1,1))
    # Cria o modelo KMeans com o número de clusters desejado
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=5)
    # Treina o KMeans nos pixels da imagem
    kmeans.fit(vectorized)
    # Obtém os valores centrais (intensidade média de cada grupo)
    centers = np.uint8(kmeans.cluster_centers_)
    # Substitui cada pixel pelo valor do centro do seu grupo (segmentação)
    segmented_data = centers[kmeans.labels_.flatten()]
    # Reconstrói a imagem segmentada no formato original
    segmented_image = segmented_data.reshape((img.shape))
    return segmented_image # Retorna a imagem segmentada

# Define o número de clusters (regiões) para segmentação
clusters = 3

# Aplica o filtro de segmentação KMeans em cada imagem
img_G_segmentada = filtro_kmeans(img_G, clusters) # Segmenta a imagem G
img_D_segmentada = filtro_kmeans(img_D, clusters) # Segmenta a imagem D
img_F_segmentada = filtro_kmeans(img_F, clusters) # Segmenta a imagem F

# Plota as imagens segmentadas para visualização
fig, axs = plt.subplots(1, 3, figsize=(10, 3)) # Cria uma figura com 3 subplots
im1 = axs[0].imshow(img_G_segmentada, cmap='gray', vmin=0, vmax=255) # Mostra a imagem G segmentada
axs[0].set_title('Segmentada G') # Título do subplot
im2 = axs[1].imshow(img_D_segmentada, cmap='gray', vmin=0, vmax=255) # Mostra a imagem D segmentada
axs[1].set_title('Segmentada D') # Título do subplot
im3 = axs[2].imshow(img_F_segmentada, cmap='gray', vmin=0, vmax=255) # Mostra a imagem F segmentada
axs[2].set_title('Segmentada F') # Título do subplot
plt.show() # Exibe as imagens segmentadas

# ----------------------------------------------------------------------
# PASSO A PASSO (MANUAL DE CLUSTERIZAÇÃO DE IMAGENS):

# 1. Instale as bibliotecas necessárias (numpy, matplotlib, scikit-learn, pillow).
# 2. Importe as bibliotecas para manipulação, visualização e clusterização de imagens.
# 3. Carregue as imagens que deseja segmentar (preferencialmente em escala de cinza).
# 4. Visualize as imagens originais para entender o padrão dos tecidos e regiões.
# 5. Defina uma função que aplique o KMeans nos pixels da imagem, agrupando-os em regiões de intensidade semelhante.
#    - O KMeans agrupa pixels com valores próximos, criando regiões homogêneas.
#    - O número de clusters define quantas regiões distintas você quer separar na imagem.
# 6. Aplique a função de segmentação em cada imagem, escolhendo o número de clusters conforme o contraste desejado.
# 7. Plote as imagens segmentadas para visualizar o resultado. As regiões da imagem agora aparecem com intensidades agrupadas,
#    facilitando a identificação de áreas de interesse (como possíveis tumores ou tecidos diferentes).
# 8. Ajuste o número de clusters para mais ou menos detalhes na segmentação, conforme necessário.