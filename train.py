from preprocess import data_loader_func
import torch
import torch.nn as nn
from model import create_model



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

train_loader, val_loader, test_loader = data_loader_func()

model = create_model()
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)






def train(model, loader, val_loader, loss_function, optimizer, device, num_epochs = 15):
    
    best_val_acc = 0.0

    for epochs in  range (num_epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)
            loss = loss_function(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            


        avg_loss = total_loss / len(loader)
        val_accuracy = evaluate(model, val_loader, device)
        print("Epoch", epochs + 1, "| avg loss:", avg_loss, "| val accuracy:", val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved")

    return best_val_acc



def evaluate(model, loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            pred = torch.argmax(outputs, dim = 1)

            correct = correct + (pred == labels).sum().item()
            total = total + labels.size(0)

    acc = correct / total
    return acc        

if __name__ == "__main__":

    best_accuracy = train(model, train_loader, val_loader, loss_function, optimizer, device)
    print("Best validation accuracy:", best_accuracy)

    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)

    test_accuracy = evaluate(model, test_loader, device)
    print("Test accuracy:", test_accuracy)
