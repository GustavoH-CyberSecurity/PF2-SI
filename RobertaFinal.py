import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Função para ler dados
def read_data(filepath):
    df = pd.read_excel(filepath)
    df['text'] = df['title'] + " " + df['description']
    return df

# Classe para o dataset
class AnunciosDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(label, dtype=torch.long)
        }

# Função para treinar o modelo
def train_model(model, train_loader, val_loader, device, epochs=3, grad_accum_steps=2):
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)
    loss_values = []

    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if step % 50 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")


        avg_epoch_loss = total_loss / len(train_loader)
        loss_values.append(avg_epoch_loss)
        print(f"Average Loss for Epoch {epoch}: {avg_epoch_loss}")


        evaluate_model(model, val_loader, device)

    # Salva o modelo treinado
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')

    # Plota o gráfico de perda
    plt.plot(loss_values, label='Training Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Perda.png")
    #plt.show()


def evaluate_model(model, data_loader, device):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    
    cm = confusion_matrix(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(classification_report(true_labels, predictions))

    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    #plt.savefig(f"MatrizDeConfusão = {epoch}.png")
    plt.show()

# Configuração do ambiente
def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    filepath = 'dbmBasesequalizadas.xlsx'
    df = read_data(filepath)
    train_df, val_df = train_test_split(df, test_size=0.3)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_dataset = AnunciosDataset(train_df['text'].to_numpy(), train_df['classificaçãoDadoPessoal'].to_numpy(), tokenizer)
    val_dataset = AnunciosDataset(val_df['text'].to_numpy(), val_df['classificaçãoDadoPessoal'].to_numpy(), tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    device = setup_device()
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)
    train_model(model, train_loader, val_loader, device)
