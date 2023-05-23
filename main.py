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

# Caminho para o diretório do dataset
dataset_path = "cats-breads"

# Obtém uma lista de todos os subdiretórios no diretório do dataset
class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

all_images = []
all_labels = []

# Para cada diretório de classe, carrega as imagens e cria os rótulos
for i, class_dir in enumerate(class_dirs):
    class_path = os.path.join(dataset_path, class_dir)
    class_images = [Image.open(os.path.join(class_path, filename)) for filename in os.listdir(class_path)]
    class_labels = [i] * len(class_images)  # Rótulo é o índice do diretório de classe
    all_images.extend(class_images)
    all_labels.extend(class_labels)

# Mistura as imagens e rótulos
combined = list(zip(all_images, all_labels))
random.shuffle(combined)
all_images, all_labels = zip(*combined)

# Separa os dados em conjuntos de treinamento e teste
train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.3,
                                                                        stratify=all_labels)

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


image = train_images[51]
hog_features = extract_hog_features(image)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.axis('off')
ax1.imshow(image, cmap='gray')
ax1.set_title('Imagem de treinamento')
ax2.bar(range(len(hog_features)), hog_features)
ax2.set_title('Características HOG')
plt.show()



# Calcule o HOG das imagens de treinamento e teste
train_hog_features = np.array([extract_hog_features(image) for image in train_images])
test_hog_features = np.array([extract_hog_features(image) for image in test_images])

# Ajuste os parâmetros de um modelo SVM usando as características HOG das imagens de treinamento
svm_model = svm.SVC()
svm_model.fit(train_hog_features, train_labels)

# Utilize o SVM treinado para classificar o conjunto de testes
predictions = svm_model.predict(test_hog_features)

 # Avalie a acurácia do modelo
accuracy = np.mean(predictions == test_labels)
print(f"Acurácia do modelo SVM: {accuracy}")
 