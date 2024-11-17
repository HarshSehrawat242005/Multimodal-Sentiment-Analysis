import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset_path = "dataset/multimodal_dataset.csv"
data = pd.read_csv(dataset_path)

# Visualize label distribution
plt.figure(figsize=(6, 6))
data['label'].value_counts().plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title("Label Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

