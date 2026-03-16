import torch
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, 
                             precision_score, 
                             recall_score, 
                             matthews_corrcoef)


class Train:
    FEATURE_ORDER = ('prnu', 'illumination', 'frequency')

    def __init__(self, device, loss, features=('prnu', 'illumination', 'frequency')):
        """
        Args:
            device:   torch.device to move tensors to.
            loss:     Loss function (e.g. smp DiceLoss).
            features: Active feature names — must match the combination used in DataLoader/Dataset.
        """
        self.device          = device
        self.loss            = loss
        self.features        = frozenset(f.lower() for f in features)
        self.active_features = [f for f in self.FEATURE_ORDER if f in self.features]

    def run_epoch(self, loader, model, optimizer=None, train=True):
        """
        Run one epoch of training or validation.

        The DataLoader yields batches as:
            (*feature_tensors_in_canonical_order, fused_t, mask_t)

        This method unpacks them dynamically based on the active feature set.

        Args:
            loader:    DataLoader for the split.
            model:     MBEN model instance.
            optimizer: Required when train=True; must be None or omitted for validation.
            train:     If True, updates model weights. If False, runs inference only.
        """
        if train and optimizer is None:
            raise ValueError("optimizer must be provided when train=True.")

        model.train() if train else model.eval()
        total_loss = 0.0

        all_preds = []
        all_targets = []

        context    = torch.enable_grad() if train else torch.no_grad()

        with context:
            for batch in tqdm(loader, desc='Train' if train else 'Val', leave=False):
                # Unpack: first N tensors are features (canonical order), then fused, then mask
                # batch = (feat_1, ..., feat_N, fused, mask)
                *feat_tensors, fused, masks = batch

                # Build feature_dict {name: tensor} and move everything to device
                feature_dict = {
                    feat: feat_tensors[i].to(self.device)
                    for i, feat in enumerate(self.active_features)
                }
                fused = fused.to(self.device)
                masks = masks.to(self.device)

                preds = model(feature_dict, fused)   # (B, 1, H, W)
                loss  = self.loss(preds, masks)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

                if not train:
                    probs = torch.sigmoid(preds)

                    all_preds.append(probs.detach().cpu().flatten())
                    all_targets.append(masks.detach().cpu().flatten())

        avg_loss = total_loss / len(loader)

        if train:
            return avg_loss

        # ---- Compute Validation Metrics ----
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        binary_preds = (all_preds > 0.5).astype(int)

        auc = roc_auc_score(all_targets, all_preds)
        precision = precision_score(all_targets, binary_preds, zero_division=0)
        recall = recall_score(all_targets, binary_preds, zero_division=0)
        mcc = matthews_corrcoef(all_targets, binary_preds)

        metrics = {
            "val_loss": avg_loss,
            "AUC_ROC": auc,
            "Precision": precision,
            "Recall": recall,
            "MCC": mcc
        }

        return metrics