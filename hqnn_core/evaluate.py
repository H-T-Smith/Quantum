import torch
import numpy as np

def evaluate_model(model, test_loader, device="cpu", return_preds=False, verbose=True):
    model.to(device)
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

        predictions = np.vstack(predictions)
        labels = np.vstack(labels)

        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)

        if verbose:
            print(f"✅ RMSE on test set: {rmse:.4f}")

        if return_preds:
            return rmse, predictions, labels
        return rmse