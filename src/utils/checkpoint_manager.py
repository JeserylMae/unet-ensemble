import os
import torch


class CheckpointManager:
    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_val_loss, 
                        early_stop_counter, train_losses, val_losses, val_auc,
                        val_precision, val_recall, val_mcc, path="checkpoint.pth"):
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss':        best_val_loss,
            'early_stop_counter':   early_stop_counter,
            'train_losses':         train_losses,
            'val_losses':           val_losses,
            'val_auc':              val_auc,
            'val_precision':        val_precision,
            'val_recall':           val_recall,
            'val_mcc':              val_mcc
        }, path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, model, optimizer, scheduler, path="checkpoint.pth"):
        if os.path.exists(path) and path:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch        = checkpoint.get('epoch', 0) + 1
            best_val_loss      = checkpoint.get('best_val_loss', float('inf'))
            early_stop_counter = checkpoint.get('early_stop_counter', 0)
            train_losses       = checkpoint.get('train_losses', [])
            val_losses         = checkpoint.get('val_losses', [])
            val_auc            = checkpoint.get('val_auc', [])
            val_precision      = checkpoint.get('val_precision', [])
            val_recall         = checkpoint.get('val_recall', [])
            val_mcc            = checkpoint.get('val_mcc', [])
            
            print(f"Resumed from epoch {start_epoch}")
            return start_epoch, best_val_loss, early_stop_counter, train_losses, val_losses, val_auc, val_precision, val_recall, val_mcc
        
        # No checkpoint found — return defaults
        return 0, float('inf'), 0, [], []

