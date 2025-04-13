from model import MLP
from train import train
from cifar10_loader import CIFAR10Loader

# 超参数搜索
def hyperparameter_search(loader):
    # 定义超参数搜索空间
    learning_rates = [0.1, 0.3]
    lr_decays = [0.99, 0.9]
    hidden_layer_sizes = [[2048, 2048], [2048, 1024]]
    l2_lambdas = [0, 0.001, 0.01]

    best_model = None
    best_val_acc = -1
    best_params = {}

    with open("search_log.txt", "w") as log_file:
        for lr in learning_rates:
            for lr_decay in lr_decays:
                for hidden_dims in hidden_layer_sizes:
                    for l2_lambda in l2_lambdas:
                        msg = f"Training with lr={lr}, lr_decay={lr_decay}, hidden_dims={hidden_dims}, l2_lambda={l2_lambda}"
                        print(msg)
                        log_file.write(msg + "\n")
                        log_file.flush()
                        
                        model = MLP(input_dim=3072, hidden_dims=hidden_dims, output_dim=10, activations=['relu', 'relu'])
                        hidden_str = "_".join(str(h) for h in hidden_dims)
                        save_path = f"lr{lr}_decay{lr_decay}_dims{hidden_str}_l2{l2_lambda}_best_model.pkl"
                        
                        train_losses, val_losses, train_accuracies, val_accuracies, best_epoch, best_val_acc_model_i = train(
                            model, loader, num_epochs=50, base_lr=lr, lr_decay=lr_decay, l2_lambda=l2_lambda, save_path=save_path, log_file=log_file)
                        
                        if best_val_acc_model_i > best_val_acc:
                            best_val_acc = best_val_acc_model_i
                            best_params = {
                                "lr": lr,
                                "lr_decay": lr_decay,
                                "hidden_dims": hidden_dims,
                                "l2_lambda": l2_lambda
                            }

        summary = f"Best model found with params: {best_params}, val_acc: {best_val_acc:.4f}"
        print(summary)
        log_file.write(summary + "\n")
        log_file.flush()

    return best_params


data_dir = './cifar-10-batches-py' 
cifar10_loader = CIFAR10Loader(data_dir=data_dir, val_ratio=0.1, batch_size=256)
hyperparameter_search(cifar10_loader)