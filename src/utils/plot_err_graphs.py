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
    plain56_df = pd.read_csv(
        f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/plain56/training_validation_acc.csv",
    )
    # resnet20_df = pd.read_csv(
    #     f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/resnet20/training_validation_acc.csv",
    # )
    # resnet32_df = pd.read_csv(
    #     f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/resnet32/training_validation_acc.csv",
    # )
    # resnet44_df = pd.read_csv(
    #     f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/resnet44/training_validation_acc.csv",
    # )
    # resnet56_df = pd.read_csv(
    #     f"/home/thorpe/git_repos/Capstone/results/performance_metrics/{repo_to_plot_from}/resnet56/training_validation_acc.csv",
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
    err_plain56_df = pd.DataFrame(
        {
            "Iterations": plain56_df["Epochs"] * 352,
            "TrainingErrorRate": 100 - plain56_df["TrainingAccuracy"],
            "ValidationErrorRate": 100 - plain56_df["ValidationAccuracy"],
        }
    )
    # err_resnet20_df = pd.DataFrame(
    #     {
    #         "Iterations": resnet20_df["Epochs"] * 352,
    #         "TrainingErrorRate": 100 - resnet20_df["TrainingAccuracy"],
    #         "ValidationErrorRate": 100 - resnet20_df["ValidationAccuracy"],
    #     }
    # )
    # err_resnet32_df = pd.DataFrame(
    #     {
    #         "Iterations": resnet32_df["Epochs"] * 352,
    #         "TrainingErrorRate": 100 - resnet32_df["TrainingAccuracy"],
    #         "ValidationErrorRate": 100 - resnet32_df["ValidationAccuracy"],
    #     }
    # )
    # err_resnet44_df = pd.DataFrame(
    #     {
    #         "Iterations": resnet44_df["Epochs"] * 352,
    #         "TrainingErrorRate": 100 - resnet44_df["TrainingAccuracy"],
    #         "ValidationErrorRate": 100 - resnet44_df["ValidationAccuracy"],
    #     }
    # )
    # err_resnet56_df = pd.DataFrame(
    #     {
    #         "Iterations": resnet56_df["Epochs"] * 352,
    #         "TrainingErrorRate": 100 - resnet56_df["TrainingAccuracy"],
    #         "ValidationErrorRate": 100 - resnet56_df["ValidationAccuracy"],
    #     }
    # )

    plt.figure(figsize=(10, 6))

    data_columns = [col for col in err_plain20_df.columns if col != "Iterations"]

    plt.axhline(y=10, color="black", linestyle="--", linewidth=0.5)
    plt.axhline(y=20, color="black", linestyle="--", linewidth=0.5)

    for data_column in data_columns:
        line_thickness = 1
        linestyle = "-"

        if data_column == "ValidationErrorRate":
            line_thickness = 2
        if data_column == "TrainingErrorRate":
            linestyle = "--"
        plt.plot(
            err_plain20_df["Iterations"],
            err_plain20_df[data_column],
            label=data_column + " Plain20",
            linewidth=line_thickness,
            linestyle=linestyle,
        )
        plt.plot(
            err_plain32_df["Iterations"],
            err_plain32_df[data_column],
            label=data_column + " Plain32",
            linewidth=line_thickness,
            linestyle=linestyle,
        )
        plt.plot(
            err_plain44_df["Iterations"],
            err_plain44_df[data_column],
            label=data_column + " Plain44",
            linewidth=line_thickness,
            linestyle=linestyle,
        )
        plt.plot(
            err_plain56_df["Iterations"],
            err_plain56_df[data_column],
            label=data_column + " Plain56",
            linewidth=line_thickness,
            linestyle=linestyle,
        )
        # plt.plot(
        #     err_resnet20_df["Iterations"],
        #     err_resnet20_df[data_column],
        #     label=data_column + "Resnet20",
        #     linewidth=line_thickness,
        #     linestyle=linestyle,
        # )
        # plt.plot(
        #     err_resnet32_df["Iterations"],
        #     err_resnet32_df[data_column],
        #     label=data_column + " Resnet32",
        #     linewidth=line_thickness,
        #     linestyle=linestyle,
        # )
        # plt.plot(
        #     err_resnet44_df["Iterations"],
        #     err_resnet44_df[data_column],
        #     label=data_column + " Resnet44",
        #     linewidth=line_thickness,
        #     linestyle=linestyle,
        # )
        # plt.plot(
        #     err_resnet56_df["Iterations"],
        #     err_resnet56_df[data_column],
        #     label=data_column + " Resnet56",
        #     linewidth=line_thickness,
        #     linestyle=linestyle,
        # )

    plt.title("Brain Dataset CNN Error Rates")
    plt.xlabel("Iterations")
    plt.ylabel("Error (%)")
    plt.ylim(0, 60)
    plt.xlim(0, 60000)
    plt.legend()
    # plt.show()

    plt.savefig(f"/home/thorpe/git_repos/Capstone/{output_name}.png")


if __name__ == "__main__":
    # plot_err_graphs("cifar", "cifar_all_plain_error_rate")
    # plot_err_graphs("brain", "brain_all_resnet_error_rate")
    plot_err_graphs("early_stop", "brain_all_cnn_error_rate")
