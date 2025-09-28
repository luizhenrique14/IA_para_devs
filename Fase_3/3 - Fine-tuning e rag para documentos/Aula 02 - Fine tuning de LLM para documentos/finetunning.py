#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemplo genérico de Fine-Tuning em Python
Tudo em um único arquivo:
 - Instalação de dependências
 - Imports
 - Logs
 - Criação de dataset simples
 - Modelo PyTorch
 - Processo de treino + avaliação
 - Salvamento do modelo
"""

import os
import subprocess
import logging
import random
import sys
import copy
import math

# ====================================================
# Instalação das dependências (caso não existam)
# ====================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    subprocess.check_call(["pip", "install", "torch"])
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset


# ====================================================
# Configuração de Logs
# ====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Simple lexical sentiment lexicon for didactic features
POSITIVE_WORDS = {"bom", "excelente", "gostei", "recomendo", "ótimo", "surpreendeu", "muito", "adoro", "ótima"}
NEGATIVE_WORDS = {"horrível", "péssima", "não", "nunca", "ruim", "terrível", "odeio", "detestei"}


def extract_features_from_text(text):
    """Return a torch tensor with two features: positive_count and negative_count (normalized).

    This is a very small, didactic feature extractor that will correlate better
    with the toy labels used in this example than length/uppercase counts.
    """
    words = [w.strip('.,!?').lower() for w in text.split()]
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = max(1, len(words))
    return torch.tensor([pos / total, neg / total], dtype=torch.float32)


# ====================================================
# Dataset de Exemplo (Simples)
# ====================================================
class SimpleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        features = extract_features_from_text(text)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


# ====================================================
# Modelo simples (Rede Neural)
# ====================================================
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ====================================================
# Fine-Tuning (treino e avaliação)
# ====================================================
def train(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # accuracy for the batch
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        epoch_acc = (total_correct / total_samples) if total_samples else 0.0
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Acc: {epoch_acc:.2%}")


def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # also collect per-sample info to make the validation behavior transparent
        for batch_idx, (features, labels) in enumerate(dataloader):
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # log per-sample predictions inside the batch
            for i in range(labels.size(0)):
                pred_label = int(predicted[i].item())
                prob0 = float(probs[i, 0].item())
                prob1 = float(probs[i, 1].item())
                logger.info(f"Val sample batch {batch_idx} idx {i}: pred={pred_label}, probs=[{prob0:.3f},{prob1:.3f}], true={int(labels[i].item())}")
    acc = correct / total
    logger.info(f"Acurácia na avaliação: {acc:.2%}")
    return acc


def predict(model, text):
    """Given a model and a single text, return logits, probabilities and predicted label.

    This helper uses the same feature extraction as the Dataset.
    Returns a dict: {'text': text, 'logits': tensor, 'probs': tensor, 'pred': int}
    """
    model.eval()
    # use the same lexical feature extractor as the dataset
    features = extract_features_from_text(text)
    # add batch dim
    with torch.no_grad():
        outputs = model(features.unsqueeze(0))
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
    return {
        'text': text,
        'logits': outputs.squeeze(0),
        'probs': probs.squeeze(0),
        'pred': int(pred.item())
    }


# ====================================================
# Pipeline Principal
# ====================================================
if __name__ == "__main__":
    logger.info("Iniciando processo de Fine-Tuning...")

    # Deterministic behavior for reproducibility in small example
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Criando um dataset maior e balanceado (positivo/negativo)
    positive_texts = [
        "Este produto é excelente",
        "Muito bom, recomendo",
        "Gostei bastante",
        "Excelente qualidade",
        "Produto ótimo, adorei",
        "Funciona muito bem",
        "Super recomendo",
        "Fiquei satisfeito",
        "Perfeito para mim",
        "Muito feliz com a compra",
        "Ótima qualidade",
        "Surpreendeu positivamente",
        "Produto fantástico",
        "Me impressionou",
        "Acabamento excelente",
        "Altamente recomendado",
        "Atendeu às expectativas",
        "Entrega rápida e boa",
        "Muito útil",
        "Produto confiável"
    ]

    negative_texts = [
        "Horrível, nunca mais compro",
        "Péssima qualidade",
        "Não gostei",
        "Muito ruim",
        "Que decepção",
        "Produto quebrado",
        "Não funciona",
        "Perdi meu dinheiro",
        "Atendimento terrível",
        "Reclamo e não resolvem",
        "Muito insatisfeito",
        "Nunca mais compro dessa marca",
        "Decepcionante",
        "Não recomendo",
        "Produto com defeito",
        "Chegou danificado",
        "Péssimo acabamento",
        "Desisti da compra",
        "Falhou no primeiro uso",
        "Ruim demais"
    ]

    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    # Stratified split: preserve class balance in train/val
    from torch.utils.data import Subset
    indices_pos = [i for i, l in enumerate(labels) if l == 1]
    indices_neg = [i for i, l in enumerate(labels) if l == 0]

    random.shuffle(indices_pos)
    random.shuffle(indices_neg)

    val_ratio = 0.25
    val_pos_count = max(1, int(len(indices_pos) * val_ratio))
    val_neg_count = max(1, int(len(indices_neg) * val_ratio))

    val_indices = indices_pos[:val_pos_count] + indices_neg[:val_neg_count]
    train_indices = indices_pos[val_pos_count:] + indices_neg[val_neg_count:]

    # shuffle train indices
    random.shuffle(train_indices)

    dataset = SimpleDataset(texts, labels)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    # collect validation texts/labels for held-out demonstration
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    # Modelo
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Fine-Tuning
    # If user provided a demo text as a CLI argument, use it to show a single-input demo
    demo_text = None
    if len(sys.argv) > 1:
        demo_text = sys.argv[1]

    # Prepare an intentionally biased copy of the model to demonstrate a wrong baseline
    # (we don't modify the real model used for training)
    def make_biased_model(m):
        bm = copy.deepcopy(m)
        # zero the first-layer weights so output depends only on bias
        with torch.no_grad():
            if hasattr(bm, 'fc1'):
                bm.fc1.weight.data.zero_()
                bm.fc1.bias.data.zero_()
            if hasattr(bm, 'fc2'):
                # bias towards class 0 (negative) so many positive samples will be mispredicted
                bm.fc2.bias.data[:] = torch.tensor([3.0, -3.0])
        return bm

    # Baseline: evaluate untrained model on validation (held-out) samples
    logger.info("--- Baseline (pré-treinamento) - previsões em amostras de validação ---")
    for t, true in zip(val_texts, val_labels):
        out = predict(model, t)
        logger.info(f"VAL PRE - Input: '{out['text']}' -> probs={out['probs'].tolist()}, pred={out['pred']}, true={true}")
    baseline_acc = evaluate(model, val_loader)

    # If demo text supplied, show intentionally wrong prediction from biased model
    if demo_text is not None:
        logger.info("--- Demonstração de input único (antes do fine-tuning) ---")
        biased = make_biased_model(model)
        b_out = predict(biased, demo_text)
        logger.info(f"DEMO BASELINE (viés intencional) - Input: '{b_out['text']}' -> probs={b_out['probs'].tolist()}, pred={b_out['pred']} (esperado: errar)")

    # Train
    train(model, train_loader, criterion, optimizer, epochs=12)
    val_acc = evaluate(model, val_loader)

    # Post-training: show same held-out validation samples again
    logger.info("--- Pós-treinamento - mesmas amostras de validação ---")
    for t, true in zip(val_texts, val_labels):
        out = predict(model, t)
        logger.info(f"VAL POST - Input: '{out['text']}' -> probs={out['probs'].tolist()}, pred={out['pred']}, true={true}")
    logger.info(f"Baseline acc: {baseline_acc:.2%} | Pós-treinamento acc: {val_acc:.2%}")

    # If demo text supplied, show prediction after training
    if demo_text is not None:
        logger.info("--- Demonstração de input único (após fine-tuning) ---")
        t_out = predict(model, demo_text)
        logger.info(f"DEMO PÓS-TRAIN - Input: '{t_out['text']}' -> probs={t_out['probs'].tolist()}, pred={t_out['pred']}")

    # Salvando modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/simple_finetuned.pt")
    logger.info("Modelo salvo em models/simple_finetuned.pt")

    # ---------------------------
    # Simple runtime tests / checks
    # ---------------------------
    logger.info("Executando testes simples...")
    tests_passed = True

    # Test 1: dataset length
    if len(dataset) != len(texts):
        logger.error("Teste falhou: tamanho do dataset diferente do esperado")
        tests_passed = False

    # Test 2: predict output shapes
    p = predict(model, "Teste rápido")
    if not (hasattr(p['logits'], 'shape') and hasattr(p['probs'], 'shape')):
        logger.error("Teste falhou: predict não retornou tensores válidos")
        tests_passed = False

    # Test 3: val_acc sanity
    if not (0.0 <= val_acc <= 1.0):
        logger.error("Teste falhou: acurácia de validação fora do intervalo esperado")
        tests_passed = False

    if tests_passed:
        logger.info("Todos os testes simples passaram ✅")
    else:
        logger.warning("Alguns testes simples falharam — verifique os logs acima")
