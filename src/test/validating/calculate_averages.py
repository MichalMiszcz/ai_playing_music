import numpy as np
import pandas as pd
import dataframe_image as dfi
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_stats(df, output_file, output_img):
    stats = df.agg(['mean', 'max'])
    # stats.loc['cv'] = stats.loc['std'] / stats.loc['mean']
    df_results = stats.reset_index()
    df_results = df_results.rename(columns={'index': 'Rodzaj statystyki'})
    name_map = {
        'mean': 'Średnia',
        # 'std': 'Odch. Std.',
        # 'median': 'Mediana',
        # 'min': 'Min',
        'max': 'Max'
        # 'cv': 'CV'
    }

    df_results['Rodzaj statystyki'] = df_results['Rodzaj statystyki'].replace(name_map)

    df_results.to_csv(output_file, index=False)

    # styled_df = df_results.style.format(precision=2).hide(axis="index")
    # dfi.export(styled_df, output_img)

    return df_results

def make_plot(df, name, output_plot_template):
    for i in range(1, 3):
        plt.figure(figsize=(10, 6))

        col = df.columns[i]

        sns.barplot(x=df.index, y=df[col])
        avg = df[col].mean()

        plt.axhline(avg, color='red', linestyle='--', label=f'Średnia: {avg:.2f}')
        plt.legend()

        plt.xlabel('Indeks')
        plt.ylabel('Wartość')

        output = output_plot_template + f'_{col}.png'

        plt.savefig(output, bbox_inches='tight', dpi=150)

def calculate():
    names = ['audiveris', 'my_model', 'scan2notes', 'sheet_vision']
    my_names = ['Audiveris', 'Mój model', 'Scan2Notes', 'Sheet Vision']
    resolutions = ['hi', 'low']
    metrics = ['DTW score', 'Levenstein score' ,'Frechet score']

    summary = {}
    for metric in metrics:
        summary[metric] = {}
        for resolution in resolutions:
            summary[metric][resolution] = pd.DataFrame()

    i = 0
    for name in names:
        for res in resolutions:
            if not (name == "sheet_vision" and res == "low"):
                file_template = f'src/csv/notes_stats_{name}_{res}.csv'
                file_template_png = f'src/csv/notes_stats_{name}_{res}.png'
                output_file_template = f'src/csv/statistics/notes_stats_{name}_{res}_statistics.csv'
                output_img_template = f'src/csv/statistics/notes_stats_{name}_{res}_statistics.png'
                # output_plot_template = f'src/csv/plots/notes_stats_{name}_{res}_plot'

                df = pd.read_csv(file_template)
                df = df[metrics]
                styled_df = df.style.format(precision=2)
                dfi.export(styled_df, file_template_png)

                df_results = calculate_stats(df, output_file_template, output_img_template)
                # make_plot(df, name, output_plot_template)

                for metric in metrics:
                    df_temp = df_results.set_index("Rodzaj statystyki")
                    summary[metric][res][my_names[i]] = df_temp[metric]

        i += 1

    for metric in metrics:
        for resolution in resolutions:
            summary[metric][resolution].to_csv(f'src/csv/statistics/summary_{metric}_{resolution}.csv')


if __name__ == '__main__':
    calculate()
