from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
model = MultimodalModel()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Replace with real preprocessing logic
    image = torch.tensor([...])
    audio = torch.tensor([...])
    with torch.no_grad():
        prediction = model(image, audio)
        _, predicted_class = torch.max(prediction, 1)
    return jsonify({'sentiment': int(predicted_class.item())})

if __name__ == '__main__':
    app.run(debug=True)
