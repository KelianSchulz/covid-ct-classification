import numpy as np
import pandas as pd
import glob
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


#Daten aus data Ordner holen und Anzahl herausfinden

dataset = Path("data")

ordner_list = []
labels = []
for ordner in dataset.iterdir():
    ordner_list.append(ordner.name)

covid_data = glob.glob(f"data/{ordner_list[1]}/*")
non_covid_data = glob.glob(f"data/{ordner_list[0]}/*")

covid_data.sort()
non_covid_data.sort()


def show_images():
    # 10 Bildern aus jeweils non und covid plotten

    # Covid Plot
    plt.figure(figsize=(20, 8))
    plt.title("COVID")
    plt.axis("off")
    for i in range (10):
        img = Image.open(covid_data[i])
        plt.subplot(2, 5, i + 1)
        plt.axis("off")
        plt.imshow(img, cmap = "gray")
    
    plt.show()


    # Non Covid Plot
    plt.figure(figsize=(20, 8))
    plt.axis("off")
    plt.title("NON COVID")
    for i in range (10):
        img = Image.open(non_covid_data[i])
        plt.subplot(2, 5, i + 1)
        plt.axis("off")
        plt.imshow(img, cmap = "gray")
    plt.show()


# Daten Labeln.
#Ordner_list[1] = covid, [0] = non_covid
alle_dateien = []
for datei in covid_data:
    alle_dateien.append(datei)
    labels.append(1)


for datei in non_covid_data:
    alle_dateien.append(datei)
    labels.append(0)


# Größe der Bilder Überprüfen
# Covid
size_covid = []
for datei in covid_data:
    bild = Image.open(datei)
    size_covid.append(bild.size)


# Non_Covid    

size_non_covid = []
for datei in non_covid_data:
    bild = Image.open(datei)
    size_non_covid.append(bild.size)

#Train, test split

from sklearn.model_selection import train_test_split

data_train, data_rest, labels_train, labels_rest = train_test_split(

alle_dateien,
labels,
test_size= 0.3,
stratify=labels,
random_state=42

)


data_val, data_test, labels_val, labels_test = train_test_split(

data_rest,
labels_rest,
test_size=0.5,
stratify=labels_rest,
random_state=42


)



def data_loader_func():
    from torchvision import torch as t, transforms


    transforms_train = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transforms_eval = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    from torch.utils.data import Dataset, DataLoader
    from PIL import Image


    class CTDataset (Dataset):

        def __init__(self, dateien, labels, transform):
            self.dateien = dateien
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.dateien)


        def __getitem__(self, index):
            pfad = self.dateien[index]
            bild = Image.open(pfad)
            bild = bild.convert("RGB")
            bild = self.transform(bild)
            label = self.labels[index]
            return bild, label




    train_dataset = CTDataset(data_train, labels_train, transforms_train)
    val_dataset = CTDataset(data_val, labels_val, transforms_eval)
    test_dataset = CTDataset(data_test, labels_test, transforms_eval)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)        



    test_bild, test_label = train_dataset[0]
    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    print(ordner_list)
    print(len(covid_data))
    print(len(non_covid_data))
    print(len(alle_dateien))
    print(len(labels))
    print(Counter(size_covid).most_common(5))
    print(Counter(size_non_covid).most_common(5))