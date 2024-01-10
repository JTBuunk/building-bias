import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# replace with own datasets
file_names = ["path_to_file",
              "path_to_file",
              "path_to_file",
              "path_to_file",
              "path_to_file"
]

# change names if neccessary
dataset_names = ['Bing', 'Google', 'Google_conf', 'OSM', 'OMF']
file_to_dataset = dict(zip(file_names, dataset_names))


# read the Excel files and add an identifier column to each
dataframes = []
for file_name in file_names:
    df = pd.read_excel(file_name)
    df['Dataset'] = file_to_dataset[file_name]  # adding a new column to identify the dataset with a custom name
    dataframes.append(df)

# combine the datasets
combined_data = pd.concat(dataframes, ignore_index=True)


def fnr_by_threshold(data, variable, thresholds):
    data = data.copy()  # avoid SettingWithCopyWarning

    # check if thresholds are provided for the variable
    if variable not in thresholds or len(thresholds[variable]) < 2:
        print(f"Not enough thresholds provided for {variable}.")
        return None, None

    try:
        # bin data into groups based on thresholds
        labels = [f'Q{i+1}' for i in range(len(thresholds[variable])-1)]
        data['Group'] = pd.cut(data[variable], bins=thresholds[variable], labels=labels, include_lowest=True)

        # calculate FNR for each group
        # change the formula for other fairness metrics
        fnr_groups = data.groupby('Group').apply(
            lambda x: x['fp'].sum() / (x['fp'].sum() + x['tp'].sum()) if (x['fp'].sum() + x['tp'].sum()) > 0 else 0
        )

        # calculate the equality of opportunity
        max_fnr = fnr_groups.max()
        min_fnr = fnr_groups.min()
        equality_of_opportunity = min_fnr / max_fnr if min_fnr != 0 else float('inf')

        return fnr_groups.reset_index(name='False_Negative_Rate'), equality_of_opportunity

    except Exception as e:
        print(f"An error occurred while processing {variable}: {e}")
        return None, None


# change with names of sensitive variables if necessary
variables = ['Pop_dens', 'RWI', 'Rural_scale', 'Bld_size']
thresholds = {

    'Pop_dens': ['insert treshhold values here'],
    'Rural_scale': ['insert treshhold values here'],
    'RWI': ['insert treshhold values here'],
    'Bld_size': ['insert treshhold values here'],

    }


# plotting and printing equality of opportunity
sns.set_theme(style="whitegrid")

for variable in variables:
    plt.figure(figsize=(10, 6))
    plot_data = pd.DataFrame()

    for dataset_label in combined_data['Dataset'].unique():
        dataset_data = combined_data[combined_data['Dataset'] == dataset_label]
        quantile_data, equality_of_opportunity = fnr_by_threshold(dataset_data, variable, thresholds)

        if quantile_data is not None:
            quantile_data['Dataset'] = dataset_label
            plot_data = pd.concat([plot_data, quantile_data])
            print(f"{dataset_label}: {equality_of_opportunity:.2f}")


    # plot the data
    ax = sns.barplot(data=plot_data, x='Group', y='False_Negative_Rate', hue='Dataset')
    ax.grid(False, axis='y')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # set title and labels with bold font
    ax.set_title(f'{variable}', fontweight='bold', color='black')
    ax.set_ylabel('FNR', fontweight='bold', color='black')
    ax.set_xlabel('', fontweight='bold', color='black')

    # make left and bottom spines bold
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    # Manually define the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Dataset', bbox_to_anchor=(1, 1), loc='upper left', title_fontsize='medium')

    plt.subplots_adjust(right=0.35, top=0.5)
    plt.show()

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title='Dataset', bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()
