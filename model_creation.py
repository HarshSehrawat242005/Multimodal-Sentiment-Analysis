import torch
import torch.nn as nn
import numpy as np

# Load processed data
data = np.load('processed_data.npy', allow_pickle=True).item()

X_train_image = torch.tensor(data['X_train_image'], dtype=torch.float32)
X_train_audio = torch.tensor(data['X_train_audio'], dtype=torch.float32)
y_train = torch.tensor(data['y_train'].values, dtype=torch.long)

# Define model
class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.image_branch = nn.Sequential(nn.Flatten(), nn.Linear(128*128*3, 128), nn.ReLU())
        self.audio_branch = nn.Sequential(nn.Linear(13, 128), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256, 3))

    def forward(self, image, audio):
        img_out = self.image_branch(image)
        aud_out = self.audio_branch(audio)
        combined = torch.cat((img_out, aud_out), dim=1)
        return self.fc(combined)

# Train model
model = MultimodalModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train_image, X_train_audio)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

torch.save(model.state_dict(), 'trained_model.pth')
