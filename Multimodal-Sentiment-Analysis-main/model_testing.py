import torch

model = MultimodalModel()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Load test data (update with your processed data)
X_test_image = ...
X_test_audio = ...
y_test = ...

with torch.no_grad():
    predictions = model(X_test_image, X_test_audio)
    _, predicted_classes = torch.max(predictions, 1)

accuracy = (predicted_classes == y_test).float().mean()
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
