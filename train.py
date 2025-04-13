import numpy as np
import pickle
from model import MLP
from cifar10_loader import CIFAR10Loader
from plot_loss_acc import plot_loss_acc

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy_loss(logits, labels):
    m = labels.shape[0]
    p = softmax(logits)
    log_likelihood = -np.log(p[range(m), labels] + 1e-9)
    loss = np.sum(log_likelihood) / m
    grad = p
    grad[range(m), labels] -= 1
    grad /= m
    return loss, grad

def accuracy(preds, labels):
    return np.mean(preds == labels)

def train(model, loader, num_epochs=50, base_lr=0.3, lr_decay=0.9, l2_lambda=0, save_path="best_model.pkl", log_file=None):
    X_train, Y_train = loader.get_train_set()
    X_val, Y_val = loader.get_val_set()

    best_val_acc = -1
    best_epoch = -1
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        epoch_loss = []
        lr = max(base_lr * (lr_decay ** epoch), 0.01)

        for batch_idx, (X_batch, Y_batch) in enumerate(loader.get_batches(X_train, Y_train, shuffle=True)):
            logits = model.forward(X_batch)
            loss, grad = cross_entropy_loss(logits, Y_batch)
            model.backward(grad)
            model.update(lr, l2_lambda)
            epoch_loss.append(loss)

        avg_train_loss = np.mean(epoch_loss)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy(train_pred, Y_train)
        val_acc = accuracy(val_pred, Y_val)

        val_logits = model.forward(X_val)
        val_loss, _ = cross_entropy_loss(val_logits, Y_val)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)

        log_msg = (f">>> Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}\n")
        
        if log_file:
            log_file.write(log_msg)
            log_file.flush()
        else:
            print(log_msg)

    return train_losses, val_losses, train_accuracies, val_accuracies, best_epoch, best_val_acc

data_dir = './cifar-10-batches-py' 
cifar10_loader = CIFAR10Loader(data_dir=data_dir, val_ratio=0.1, batch_size=256)
hidden_dims = [2048, 1024]
model = MLP(input_dim=3072, hidden_dims=hidden_dims, output_dim=10, activations=['relu', 'relu'])
train_losses, val_losses, train_accuracies, val_accuracies, best_epoch, best_val_acc = train(model, cifar10_loader, num_epochs=50, base_lr=0.3, lr_decay=0.9, l2_lambda=0, save_path="best_model.pkl")
plot_loss_acc(train_losses, val_losses, train_accuracies, val_accuracies)


