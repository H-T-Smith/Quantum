import torch

def train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        num_epochs,
        device="cpu",
        verbose=True,
):
    model.to(device)
    model.train()

    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    return epoch_losses