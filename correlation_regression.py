import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Read Excel file
file_path = "C:\\Users\\Johnt\\Documents\\GIMA\\Internship\\phl_Bing_tile_results_try2.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# # Filter out rows where FNR is below zero
# df = df[df['precision'] >= 0]

# Example variables - replace these with your actual variable names
dependent_var = 'fnr'  # Replace with your dependent variable for regression
independent_vars = ['poverty', 'pop_dens', 'vul_pop', 'elev', 'RWI', 'urban', 'popdens_fb', 'building_dens']  # Replace with your independent variables
weight_var = 'building_dens'  # Replace with your weight variable

# Weighted Correlation Analysis
correlation_results = {}
for var in independent_vars:
    correlation, p_value = stats.pearsonr(df[var] * df[weight_var], df[dependent_var] * df[weight_var])
    correlation_results[var] = {'correlation': correlation, 'p_value': p_value, 'significant': p_value < 0.05}

# Display Correlation Results
print("Correlation Results:")
for var, results in correlation_results.items():
    significance = "Significant" if results['significant'] else "Not significant"
    print(f"{var}: Correlation = {results['correlation']:.3f}, P-value = {results['p_value']:.3f} ({significance})")

# Weighted Linear Regression Analysis
for var in independent_vars:
    X = df[[var]]
    Y = df[dependent_var]
    weights = df[weight_var]
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.WLS(Y, X, weights=weights).fit()

    # Scatter plot with trend line
    fig, ax = plt.subplots()
    ax.scatter(X[var], Y, alpha=0.5, color='black')  # Set dots to black
    ax.plot(X[var], model.predict(), color='black', linewidth=2)  # Trend line with a bit more thickness for emphasis

    # Make all text and titles bold
    ax.set_title(f"Scatter Plot for {var}", fontweight='bold')
    ax.set_xlabel(var, fontweight='bold')
    ax.set_ylabel(dependent_var, fontweight='bold')

    # Set major locator for x-axis and y-axis to show integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Set tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make left and bottom spines bold
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Add R² value and regression formula within the plot
    # Position the text box at the bottom right of the plot
    plt.text(0.95, 0.05, f'R² = {model.rsquared:.3f}\nY = {model.params[1]:.2f}x + {model.params[0]:.2f}',
             horizontalalignment='right', verticalalignment='bottom',
             transform=ax.transAxes, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


    # Print regression summary
    print(f"Regression Results for {var}:")
    if model.f_pvalue < 0.05:
        print(model.summary())
        plt.show()
    else:
        plt.show()
        print("Not significant")
