import os
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure
from sklearn import svm
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

# Diretórios contendo as imagens
persian_dir = "cats-breads\Persian"
siamese_dir = "cats-breads\Siamese"

# Lista de rótulos das raças de gatos
races = ["Persian", "Siamese"]

# Listas para armazenar as imagens e rótulos
images = []
labels = []

# Função para percorrer um diretório e atribuir rótulos
def process_directory(directory, label):
    for file_name in os.listdir(directory):
        image_path = os.path.join(directory, file_name)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            images.append(image)
            labels.append(label)

# Percorre o diretório dos gatos persas e atribui o rótulo "Persian"
process_directory(persian_dir, 0)

# Percorre o diretório dos gatos siameses e atribui o rótulo "Siamese"
process_directory(siamese_dir, 1)

# Agora você tem as imagens e os rótulos prontos para o treinamento do modelo
# Você pode usar as listas "images" e "labels" para extrair as características HOG e treinar o modelo SVM


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=42)


# Definição dos parâmetros para o cálculo das características HOG
orientations = 10
pixels_per_cell = (4, 4)
cells_per_block = (4, 4)
img_lenght = 80

# Função para extrair as características HOG de uma imagem em escala de cinza
def extract_hog_features(image):
    image = image.convert("RGB")  # Converter a imagem para o modo RGB
    image_gray = color.rgb2gray(np.array(image))  # Converter para um array numpy
    resized_image = resize(image_gray, (img_lenght, img_lenght))
    hog_features = hog(resized_image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L2-Hys')
    return hog_features

# Exibe as características HOG de uma imagem do conjunto de treinamento
image = train_images[40]
hog_features = extract_hog_features(image)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.axis('off')
ax1.imshow(image, cmap='gray')
ax1.set_title('Imagem de treinamento')
ax2.bar(range(len(hog_features)), hog_features)
ax2.set_title('Características HOG')
plt.show()



train_hog_features = np.array([extract_hog_features(image) for image in train_images])
test_hog_features = np.array([extract_hog_features(image) for image in test_images])

svm_model = svm.SVC()
svm_model.fit(train_hog_features, train_labels)
predictions = svm_model.predict(test_hog_features)

 # Avalie a acurácia do modelo
accuracy = np.mean(predictions == test_labels)
print(f"Acurácia do modelo SVM: {accuracy * 100:.2f}%")
 