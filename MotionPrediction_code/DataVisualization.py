
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("E:/zycpostgraduate/非机动车轨迹预测/data.csv")


def density_graph(df, title):

    plt.figure(figsize=(10, 8), dpi=80)
    sns.kdeplot(df.loc[df['cyl'] == 4, "cty"],
                shade=True,
                color="blue",
                label="Cyl=4",
                alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 5, "cty"],
                shade=True,
                color="green",
                label="Cyl=5",
                alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 6, "cty"],
                shade=True,
                color="red",
                label="Cyl=6",
                alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 8, "cty"],
                shade=True,
                color="purple",
                label="Cyl=8",
                alpha=.7)


    sns.set(style="whitegrid", font_scale=1.1)
    plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=18)
    plt.legend()
    plt.show()