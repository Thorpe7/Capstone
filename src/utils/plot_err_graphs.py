import pandas as pd
import matplotlib.pyplot as plt

plain18_df = pd.read_csv(
    "results/performance_metrics/plain34/training_validation_acc.csv",
)
# plain34_df = pd.read_csv("results/performance_metrics/Plain34/training_validation_acc.csv",)
err_plain18_df = pd.DataFrame(
    {
        "Iterations": plain18_df["Epochs"] * 391,
        "TrainingErrorRate": 100 - plain18_df["TrainingAccuracy"],
        "ValidationErrorRate": 100 - plain18_df["ValidationAccuracy"],
    }
)
# err_plain34_df = pd.DataFrame(
#     {
#         "Iterations": plain34_df["Epochs"] * 30,
#         "TrainingErrorRate": 100 - plain34_df["TrainingAccuracy"],
#         "ValidationErrorRate": 100 - plain34_df["ValidationAccuracy"],
#     }
# )
print(err_plain18_df.head())

plt.figure(figsize=(10, 6))

data_columns = [col for col in err_plain18_df.columns if col != "Iterations"]

for data_column in data_columns:
    line_thickness = 1

    if data_column == "ValidationErrorRate":
        line_thickness = 1
    # plt.plot(err_plain34_df["Iterations"], err_plain34_df[data_column], label=data_column+" Plain34", linewidth=line_thickness)
    plt.plot(
        err_plain18_df["Iterations"],
        err_plain18_df[data_column],
        label=data_column + " Plain18",
        linewidth=line_thickness,
    )

plt.title("Multiple Data Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Values")
plt.legend()
plt.show()

plt.savefig('plain34_err_plt.png')
