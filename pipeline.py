from src.data_preprocessing import load_data, preprocess_text_data, split_data
from src.model_creation import create_lstm_model
from src.model_testing import evaluate_model
from tensorflow.keras.models import Sequential

def main():
    # Load and preprocess data
    data = load_data("data/raw/sentiment_data.csv")
    data = preprocess_text_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    # Create and train the model
    lstm_model = create_lstm_model(input_dim=5000)  # Example input dimension
    lstm_model.fit(X_train, y_train, epochs=3, batch_size=64)

    # Evaluate the model
    evaluate_model(lstm_model, X_test, y_test, model_type="lstm")

if _name_ == "_main_":
    main()