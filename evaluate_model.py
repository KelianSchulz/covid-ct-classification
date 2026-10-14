import os
import torch
from sklearn.metrics import confusion_matrix, classification_report

from preprocess import data_loader_func
from model import create_model

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")
matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")



def get_predictions(model, loader, device):
    model.eval()
    all_predictions = []
    all_labels = []


    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            predictions_list = predictions.cpu().tolist()
            labels_list = labels.tolist()

            for prediction in predictions_list:
                all_predictions.append(prediction)

            for label in labels_list:
                all_labels.append(label)

    return all_labels, all_predictions


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_loader, val_loader, test_loader = data_loader_func()

    model = create_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    true_labels, predicted_labels = get_predictions(model, test_loader, device)

    matrix = confusion_matrix(true_labels, predicted_labels)
    print(classification_report(true_labels, predicted_labels, target_names=["non-COVID", "COVID"]))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["non-COVID", "COVID"])
    display.plot(cmap="Blues")

    plt.title("Confusion matrix (test set)")
    plt.savefig(matrix_path, dpi=150, bbox_inches="tight")
    plt.show()
    