import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES ---
IMAGE_DIR = "imagens"         # Pasta com imagens
ANNOTATION_DIR = "anotacoes"  # Pasta com XMLs do labelImg
IMAGE_SIZE = (64, 64)         # Redimensionar as imagens
CLASSES = ["pessoa", "carro"] # Classes que queremos identificar

# --- FUNÇÃO PARA CARREGAR ANOTAÇÕES DO LABELIMG ---
def load_annotations(annotation_dir):
    annotations = []
    for file in os.listdir(annotation_dir):
        if not file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(annotation_dir, file))
        root = tree.getroot()

        filename = root.find('filename').text
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text.lower()
            labels.append(label)
        annotations.append((filename, labels))
    return annotations

# --- PREPARAR DADOS ---
annotations = load_annotations(ANNOTATION_DIR)
X = []
y = []

for filename, labels in annotations:
    img_path = os.path.join(IMAGE_DIR, filename)
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    X.append(img_array)
    
    # Multilabel binário: 1 se a classe está na lista, 0 caso contrário
    label_vector = [1 if cls in labels else 0 for cls in CLASSES]
    y.append(label_vector)

X = np.array(X)
y = np.array(y)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODELO CNN SIMPLES MULTILABEL ---
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=IMAGE_SIZE+(3,)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(CLASSES), activation='sigmoid')  # Saída multilabel
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Pra multilabel
              metrics=['accuracy'])

# --- TREINAR ---
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# --- AVALIAR ---
loss, acc = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.3f}, Accuracy: {acc:.3f}")

# --- EXEMPLO DE PREVISÃO ---
# Supondo uma imagem nova:
# nova_img = load_img("imagens/teste.jpg", target_size=IMAGE_SIZE)
# nova_array = img_to_array(nova_img) / 255.0
# nova_array = np.expand_dims(nova_array, axis=0)
# pred = model.predict(nova_array)[0]
# print(f"Pessoa: {pred[0]:.2f}, Carro: {pred[1]:.2f}")
