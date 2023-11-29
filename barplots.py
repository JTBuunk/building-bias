import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# File names and custom dataset names
file_names = ["C:\\Users\\Johnt\\Documents\\GIMA\\Internship\\Bing_tile_results1.xlsx",
              "C:\\Users\\Johnt\\Documents\\GIMA\\Internship\\Google_tile_results1.xlsx",
              "C:\\Users\\Johnt\\Documents\\GIMA\\Internship\\OSM_tile_results1.xlsx"]
dataset_names = ['Bing', 'Google', 'OSM']
file_to_dataset = dict(zip(file_names, dataset_names))



def clean_numeric(column):
    # Attempt to convert to numeric, coercing errors into NaN
    return pd.to_numeric(column, errors='coerce')

# Read the Excel files and add an identifier column to each
dataframes = []
for file_name in file_names:
    df = pd.read_excel(file_name)
    df['Dataset'] = file_to_dataset[file_name]  # Adding a new column to identify the dataset with a custom name
    dataframes.append(df)

# Combine the datasets
combined_data = pd.concat(dataframes, ignore_index=True)


def fnr_by_threshold(data, variable, thresholds):
    data = data.copy()  # Avoid SettingWithCopyWarning

    # Check if thresholds are provided for the variable
    if variable not in thresholds or len(thresholds[variable]) < 2:
        print(f"Not enough thresholds provided for {variable}.")
        return None, None

    try:
        # Bin data into groups based on thresholds
        labels = [f'Q{i+1}' for i in range(len(thresholds[variable])-1)]
        data['Group'] = pd.cut(data[variable], bins=thresholds[variable], labels=labels, include_lowest=True)

        # Calculate FNR for each group
        fnr_groups = data.groupby('Group').apply(
            lambda x: x['tp'].sum() / (x['fp'].sum() + x['tp'].sum()) if (x['tp'].sum() + x['fp'].sum()) > 0 else 0
        )

        # Calculate the equality of opportunity
        max_fnr = fnr_groups.max()
        min_fnr = fnr_groups.min()
        equality_of_opportunity = min_fnr / max_fnr if min_fnr != 0 else float('inf')

        return fnr_groups.reset_index(name='False_Negative_Rate'), equality_of_opportunity

        # # Calculate average of count_rel for each group
        # avg_count_rel_groups = data.groupby('Group')['count_rel'].mean()
        # return avg_count_rel_groups.reset_index(name='Average_Count_Rel')

    except Exception as e:
        print(f"An error occurred while processing {variable}: {e}")
        return None, None


variables = ['poverty', 'Pop_dens', 'Vul_pop', 'RWI', 'urban']
thresholds = {
    'poverty': [1.27, 10.95, 20.52, 31.51, 89.55],
    'Pop_dens': [5.428,177.606,313.122,604.707,175585.941],
    'Vul_pop': [0,0.0126,0.0244,0.0376,0.2527],
    'RWI': [-1.757,-0.545,-0.307,0.071,2.287],
    'urban': [1,12,17,21,30],
    }

# Plotting and printing equality of opportunity
sns.set_theme(style="whitegrid")

def detect_outliers(data, variable):
    Q1 = data[variable].quantile(0.25)
    Q3 = data[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    outliers_bool = (data[variable] < lower_bound) | (data[variable] > upper_bound)
    outliers = data.loc[outliers_bool]
    return outliers

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

            # # Print outliers for each quantile
            # outliers = detect_outliers(quantile_data, 'False_Negative_Rate')
            # if not outliers.empty:
            #     print(f"Outliers for {variable} in {dataset_label} dataset:")
            #     print(outliers)

    # Plot the data without considering outliers
    sns.barplot(data=plot_data, x='Group', y='False_Negative_Rate', hue='Dataset')
    plt.title(f'{variable}')
    plt.ylabel('FNR')
    plt.xlabel('')

    # Manually define the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title='Dataset', bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()
