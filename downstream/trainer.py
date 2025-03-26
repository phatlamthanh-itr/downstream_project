import os
import torch
import numpy as np
from torch import nn
from downstream.downstream_config import downstream_config
from sklearn.metrics import precision_score, recall_score, confusion_matrix



def train_model(model, train_loader, train_labels, val_loader, val_labels, 
                epochs=downstream_config.epochs, batch_size=downstream_config.batch_size, 
                learning_rate=downstream_config.lr, 
                run_dir=downstream_config.run_dir,
                log_dir = downstream_config.log_dir,
                save_epochfreq=downstream_config.save_epochfreq,
                save_pred_epoch=downstream_config.save_pred_epoch,
                criterion=downstream_config.loss,
                lr_step_size=downstream_config.lr_step_size,
                lr_gamma=downstream_config.lr_gamma,
                device="cuda" if torch.cuda.is_available() else "cpu"):
    
    # Move model to device
    model.to(device)
      
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    best_val_loss = np.inf
    best_val_acc = 0
    
    log_file = f"{log_dir}/training.out"
    with open(log_file, "w") as f:
        f.write("Epoch, Train Loss, Val Loss, Val Acc,Precision,Recall\n")
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for idx, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0], batch[1]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch, x_batch)  
            output = output.squeeze(-1)  
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        if epoch % save_pred_epoch == 0:
            val_loss, val_acc, precision, recall, conf_matrix = eval_model(model, val_loader, criterion, device, save_pred=True, save_pred_dir=downstream_config.save_pred_dir)
        else:
            val_loss, val_acc, precision, recall, conf_matrix = eval_model(model, val_loader, criterion, device)
            
        scheduler.step()
        
        state_dict = {
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
    
        if epoch % save_epochfreq == 0:
            torch.save(state_dict, f'{run_dir}/checkpoint_epoch{epoch}.pkl')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(state_dict, f'{run_dir}/checkpoint_best.pkl')
            print("Save the best checkpoint at epoch", epoch, f'{run_dir}/checkpoint_best.pkl')
        torch.save(state_dict, f'{run_dir}/checkpoint_latest.pkl')
        
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{val_loss:.4f},{val_acc:.4f},{precision:.4f},{recall:.4f}\n")

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}", "Confusion matrix:\n", conf_matrix)
    
    return best_val_loss, best_val_acc



def eval_model(model, val_loader, criterion, device, save_pred = False, save_pred_dir = downstream_config.save_pred_dir):
    model.eval()  
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch[0], batch[1]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch, x_batch)
            output = output.squeeze(-1)
            loss = criterion(output, y_batch)
            total_loss += loss.item()

            # Convert logits to binary predictions
            preds = (torch.sigmoid(output) > 0.5).float()   
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()

    # Compute average loss
    avg_loss = total_loss / len(val_loader)

    # Concatenate predictions and labels
    preds_to_save = all_preds.copy()
    labels_to_save = all_labels.copy()
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Save model predictions
    if save_pred:
        np.save(f'{save_pred_dir}/'"val_predictions.npy", np.concatenate(preds_to_save))
        np.save(f'{save_pred_dir}/'"val_labels.npy", np.concatenate(labels_to_save))

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    accuracy = correct / total
    return avg_loss, accuracy, precision, recall, conf_matrix