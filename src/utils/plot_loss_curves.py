import pandas as pd
import matplotlib.pyplot as plt

loss_df = pd.read_csv("results/performance_metrics/training_validation_loss.csv")

plt.figure(figsize=(10, 6))

data_columns = [col for col in loss_df.columns if col != "Epochs"]

for data_column in data_columns:
    plt.plot(
        loss_df["Epochs"],
        loss_df[data_column],
        label=data_column + "Plain18",
    )

plt.ylim(0, 8)
plt.title("Training and Validation Loss Vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
