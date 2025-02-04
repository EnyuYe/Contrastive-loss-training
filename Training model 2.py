import json
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_processing_counterfact import create_dataset_pairs, get_data_loader
import torch.nn.functional as F


# Step 1: Load and preprocess dataset
def load_jsonl(file_path, limit=None):
    """
    Load a JSONL file and skip invalid or empty lines.
    """
    dataset = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {i}: {line} ({e})")
    return dataset


# Step 2: Define neural network model
class SimpleNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.norm5 = nn.LayerNorm(hidden_dim)
        self.relu5 = nn.ReLU()

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, apply_softmax=False):
        x = self.relu1(self.norm1(self.fc1(x)))
        x = self.relu2(self.norm2(self.fc2(x)))
        x = self.relu3(self.norm3(self.fc3(x)))
        x = self.relu4(self.norm4(self.fc4(x)))

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        logits = self.fc_out(x)
        return logits



# Step 5: Define Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        # - label = 0, 1 → binary_label = 0 (positive)
        # - label = 2 → binary_label = 1 (negative)
        binary_label = (label == 2).float()

        loss = torch.mean((1 - binary_label) * torch.pow(euclidean_distance, 2) +
                          binary_label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss


# Step 6: Train the model
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10, device="cuda"):
    model.train()

    epoch_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        batch_count = 0

        for emb1, emb2, labels, *_ in train_loader:
            batch_count += 1

            emb1 = emb1.to(device, dtype=torch.float)
            emb2 = emb2.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            output1 = model(emb1)
            output2 = model(emb2)

            loss = criterion(output1, output2, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / batch_count
        epoch_losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.5f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker="o", linestyle="-", color="b", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return epoch_losses




def test_distances(model, test_loader, margin_list=[0.5, 1.0, 1.5, 2.0], device="cuda"):
    model.eval()
    accuracy_list = []

    with torch.no_grad():
        for margin in margin_list:
            total_samples = 0
            correct_predictions = 0

            for batch_idx, (emb1, emb2, labels, *_) in enumerate(test_loader):
                emb1 = emb1.to(device, dtype=torch.float)
                emb2 = emb2.to(device, dtype=torch.float)
                labels = labels.to(device)

                output1 = model(emb1)
                output2 = model(emb2)

                distances = F.pairwise_distance(output1, output2)

                predicted_labels = torch.where(distances < margin, torch.tensor(1, device=device), torch.tensor(2, device=device))

                labels = torch.where(labels == 0, torch.tensor(1, device=device), labels)

                correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += labels.numel()

            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            accuracy_list.append(accuracy)

            print(f"Margin: {margin:.5f} - Accuracy: {accuracy:.4f}")

    return margin_list, accuracy_list





# main process
if __name__ == "__main__":
    # File path to your JSONL dataset
    dataset_path = "counterfact_test_2_lama_merged.jsonl"

    # Step 1: Load dataset and generate embeddings
    print("Start loading dataset...")
    dataset = load_jsonl(dataset_path)


    # Step 2: Create pairs
    print("Creating dataset pairs...")
    (openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict,
     neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict,
     paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test) = create_dataset_pairs(dataset, neightbour_control=0,label_reversal=False)


    # Step 3: Create DataLoader
    num_batches = 400
    train_batch_size = len(paraphrase_train_vectors_dict) // num_batches
    test_batch_size = len(paraphrase_test_vectors_dict) // num_batches

    train_loader = get_data_loader(dataset_paired_train, openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict,paraphrase_train_vectors_dict, batch_size=train_batch_size, shuffle=True)
    test_loader = get_data_loader(dataset_paired_test, openai_vectors_dict, edit_vectors_dict, neighbourhood_test_vectors_dict,paraphrase_test_vectors_dict, batch_size=test_batch_size, shuffle=True)

    print(f"Train batch size: {train_batch_size}, Test batch size: {test_batch_size}")
    print(f"Total samples in test set: {len(paraphrase_test_vectors_dict)}")




    #Step 4: Initializing model
    print("Initializing model...")
    input_dim = 4096
    hidden_dim = 128
    output_dim = 4999

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleNet(input_dim, hidden_dim, output_dim).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


    # Step 6: Train the model
    print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=3, device=device)


    # Step 7: Test model
    # Evaluating model on test set
    print("Evaluating model on test set...")
    margin_values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    margins, accuracies = test_distances(model, test_loader, margin_values, device=device)

    # Step 8: Plot
    plt.plot(margins, accuracies, marker='o', linestyle='-')
    plt.xlabel("Margin")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Margin")
    plt.grid()
    plt.show()