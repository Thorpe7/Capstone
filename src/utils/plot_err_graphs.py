import pandas as pd
import matplotlib.pyplot as plt


def plot_err_graphs(repo_to_plot_from: str, output_name: str) -> None:
    plain20_df = pd.read_csv(
        f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/plain20/training_validation_acc.csv",
    )
    plain32_df = pd.read_csv(
        f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/plain32/training_validation_acc.csv",
    )
    plain44_df = pd.read_csv(
        f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/plain44/training_validation_acc.csv",
    )
    # resnet18_df = pd.read_csv(
    #     f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/resnet18/training_validation_acc.csv",
    # )
    # resnet34_df = pd.read_csv(
    #     f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/resnet34/training_validation_acc.csv",
    # )
    # resnet50_df = pd.read_csv(
    #     f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/resnet50/training_validation_acc.csv",
    # )
    err_plain20_df = pd.DataFrame(
        {
            "Iterations": plain20_df["Epochs"] * 352,
            "TrainingErrorRate": 100 - plain20_df["TrainingAccuracy"],
            "ValidationErrorRate": 100 - plain20_df["ValidationAccuracy"],
        }
    )
    err_plain32_df = pd.DataFrame(
        {
            "Iterations": plain32_df["Epochs"] * 352,
            "TrainingErrorRate": 100 - plain32_df["TrainingAccuracy"],
            "ValidationErrorRate": 100 - plain32_df["ValidationAccuracy"],
        }
    )
    err_plain44_df = pd.DataFrame(
        {
            "Iterations": plain44_df["Epochs"] * 352,
            "TrainingErrorRate": 100 - plain44_df["TrainingAccuracy"],
            "ValidationErrorRate": 100 - plain44_df["ValidationAccuracy"],
        }
    )
    # err_resnet18_df = pd.DataFrame(
    #     {
    #         "Iterations": resnet18_df["Epochs"] * 352,
    #         "TrainingErrorRate": 100 - resnet18_df["TrainingAccuracy"],
    #         "ValidationErrorRate": 100 - resnet18_df["ValidationAccuracy"],
    #     }
    # )
    # err_resnet34_df = pd.DataFrame(
    #     {
    #         "Iterations": resnet34_df["Epochs"] * 352,
    #         "TrainingErrorRate": 100 - resnet34_df["TrainingAccuracy"],
    #         "ValidationErrorRate": 100 - resnet34_df["ValidationAccuracy"],
    #     }
    # )
    # err_resnet50_df = pd.DataFrame(
    #     {
    #         "Iterations": resnet50_df["Epochs"] * 391,
    #         "TrainingErrorRate": 100 - resnet50_df["TrainingAccuracy"],
    #         "ValidationErrorRate": 100 - resnet50_df["ValidationAccuracy"],
    #     }
    # )

    plt.figure(figsize=(10, 6))

    data_columns = [col for col in err_plain44_df.columns if col != "Iterations"]

    for data_column in data_columns:
        line_thickness = 1

        if data_column == "ValidationErrorRate":
            line_thickness = 2
        plt.plot(
            err_plain20_df["Iterations"],
            err_plain20_df[data_column],
            label=data_column + " Plain20",
            linewidth=line_thickness,
        )
        plt.plot(
            err_plain32_df["Iterations"],
            err_plain32_df[data_column],
            label=data_column + " Plain32",
            linewidth=line_thickness,
        )
        plt.plot(
            err_plain44_df["Iterations"],
            err_plain44_df[data_column],
            label=data_column + " Plain44",
            linewidth=line_thickness,
        )
        # plt.plot(
        #     err_resnet18_df["Iterations"],
        #     err_resnet18_df[data_column],
        #     label=data_column + " Resnet18",
        #     linewidth=line_thickness,
        # )
        # plt.plot(
        #     err_resnet34_df["Iterations"],
        #     err_resnet34_df[data_column],
        #     label=data_column + " Resnet34",
        #     linewidth=line_thickness,
        # )
        # plt.plot(
        #     err_resnet50_df["Iterations"],
        #     err_resnet50_df[data_column],
        #     label=data_column + " Resnet50",
        #     linewidth=line_thickness,
        # )

    plt.title("CIFAR-10 Error Rate Over Time")
    plt.xlabel("Iterations")
    plt.ylabel("Values")
    plt.ylim(0, 40)
    plt.xlim(0, 60000)
    plt.legend()
    # plt.show()

    plt.savefig(f"/home/thorpe/git_repos/Capstone/{output_name}.png")


if __name__ == "__main__":
    plot_err_graphs("cifar", "running_error_rate")
