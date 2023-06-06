import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from dataset import create_datasets, create_loaders, get_transforms
from model import create_model
from tqdm import tqdm


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            with open("./checkpoint.pt", "wb") as f:
                torch.save(model.state_dict(), f)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%"
        )

    model.load_state_dict(torch.load("./checkpoint.pt"))
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            predictions += predicted.tolist()
            true_labels += labels.tolist()

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * test_correct / test_total

    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")

    class_names = [
        "dyed-lifted-polyps",
        "dyed-resection-margins",
        "esophagitis",
        "normal-cecum",
        "polyps",
        "ulcerative-colitis",
    ]
    print("Confusion Matrix")
    print(confusion_matrix(true_labels, predictions))
    print("Classification Report")
    print(classification_report(true_labels, predictions, target_names=class_names))


def main(
    train_path,
    test_path,
    batch_size,
    img_height,
    img_width,
    learning_rate,
    weight_decay,
    num_epochs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform, test_transform = get_transforms(img_height, img_width)

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_path, test_path, train_transform, test_transform
    )
    train_loader, val_loader, test_loader = create_loaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    model = create_model(img_height).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs,
        device,
    )


if __name__ == "__main__":
    train_path = "kvasir-dataset/"
    test_path = "kvasir-dataset/"
    batch_size = 32
    img_height = 128
    img_width = 128
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 25

    main(
        train_path,
        test_path,
        batch_size,
        img_height,
        img_width,
        learning_rate,
        weight_decay,
        num_epochs,
    )
