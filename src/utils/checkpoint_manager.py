import os
import torch


class CheckpointManager:
    def save_checkpoint(self, model, optimizer, epoch, loss, path="checkpoint.pth"):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(model, optimizer, path="checkpoint.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            print(f"Resumed from epoch {start_epoch}")
            return start_epoch
        return 0  

