import matplotlib.pyplot as plt
import seaborn as sns


def bivariate_kde(data1, title1, data2, title2, plot_title=None):
    # https://seaborn.pydata.org/examples/multiple_joint_kde.html
    sns.set(style="darkgrid")
    f, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    if plot_title:
        plt.title(plot_title)

    plt.xlabel("Comp. 0")
    if data1.shape[0] == 1:
    # one dimension data
        ax = sns.kdeplot(data=data1[0], shade=True, shade_lowest=False)
        ax = sns.kdeplot(data=data2[0], shade=True, shade_lowest=False)
    else:
    # two dimensions
        ax = sns.kdeplot(data=data1[0], data2=data1[1], shade=True, shade_lowest=False, cmap="Blues")
        ax = sns.kdeplot(data=data2[0], data2=data2[1], shade=True, shade_lowest=False, cmap="Reds")
        plt.ylabel("Comp. 1")

        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        ax.text(1, 2, title1, size=16, color=blue)
        ax.text(1, 2.5, title2, size=16, color=red)


