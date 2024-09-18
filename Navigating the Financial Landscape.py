import pandas as pd

# Update the file path to the correct location
file_path = r'C:/Users/ioann/Downloads\financial.csv'
data = pd.read_csv(file_path)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Convert 'date' column to datetime format, handling potential mixed formats
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Handle rows where 'date' could not be converted
data = data.dropna(subset=['date'])

# Filter to include only the top 30 firms by Total_Revenue
top_firms_by_revenue = data.groupby('firm')['Total_Revenue'].sum().nlargest(30).index
filtered_data = data[data['firm'].isin(top_firms_by_revenue)]

# Relationship between Total Revenue and Net Income for the top 30 firms
plt.figure(figsize=(14, 8))
sns.scatterplot(x='Total_Revenue', y='Net_Income', hue='firm', data=filtered_data, palette='tab10')

# Customizing the plot
plt.title('Total Revenue vs Net Income for Top 30 Firms (2020)', fontsize=16)
plt.xlabel('Total Revenue', fontsize=14)
plt.ylabel('Net Income', fontsize=14)
plt.legend(title='Firm', bbox_to_anchor=(1.05, 1), loc='upper left')

# Annotate the top 3 firms by Total Revenue
top_3_firms = filtered_data.nlargest(3, 'Total_Revenue')[['firm', 'Total_Revenue', 'Net_Income']]
for i in range(top_3_firms.shape[0]):
    plt.text(x=top_3_firms['Total_Revenue'].iloc[i],
             y=top_3_firms['Net_Income'].iloc[i],
             s=top_3_firms['firm'].iloc[i],
             fontsize=12,
             fontweight='bold',
             ha='right')

plt.grid(True)

# Adjust layout to avoid warnings
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.8, hspace=0.5, wspace=0.5)

plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Convert 'date' column to datetime format, handling potential mixed formats
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Handle rows where 'date' could not be converted
data = data.dropna(subset=['date'])

# Select only numeric columns for correlation matrix
numeric_data = data.select_dtypes(include=[np.number])

# Exploratory Data Analysis (EDA)
# Summary statistics
summary_stats = numeric_data.describe()

# Correlation matrix
correlation_matrix = numeric_data.corr()

# Filter to include only the top 5 firms by Total_Revenue
top_firms_by_revenue = data.groupby('firm')['Total_Revenue'].sum().nlargest(5).index
filtered_data = data[data['firm'].isin(top_firms_by_revenue)]

# Profitability analysis for top 5 firms
profitability_data = filtered_data[['firm', 'Total_Revenue', 'Net_Income']]
profitability_summary = profitability_data.groupby('firm').sum().reset_index()

# Visualization: Total Revenue vs Net Income for top 5 firms
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar plot for Total Revenue
sns.barplot(x='firm', y='Total_Revenue', data=profitability_summary, color='blue', alpha=0.6, label='Total Revenue', ax=ax1)

# Bar plot for Net Income
sns.barplot(x='firm', y='Net_Income', data=profitability_summary, color='red', alpha=0.6, label='Net Income', ax=ax1)

# Customizing the plot
ax1.set_title('Total Revenue and Net Income for Top 5 Firms (2020)', fontsize=16)
ax1.set_xlabel('Firm', fontsize=14)
ax1.set_ylabel('Amount (in Billions)', fontsize=14)
ax1.legend()

plt.show()

# Statistical test: Compare the average Net Income of the top 5 firms
net_income_top5 = filtered_data.pivot_table(values='Net_Income', index='date', columns='firm')
anova_result = stats.f_oneway(*[net_income_top5[firm].dropna() for firm in top_firms_by_revenue])

# Print ANOVA results
print('ANOVA Test Results:')
print(f'F-statistic: {anova_result.statistic:.2f}')
print(f'p-value: {anova_result.pvalue:.3f}')

# Print the comprehensive report
print("\nComprehensive Financial Analysis Report")
print("---------------------------------------")
print("1. Summary Statistics:")
print(summary_stats[['Research_Development', 'Income_Before_Tax', 'Net_Income', 'Total_Revenue']])
print("\n2. Significant Correlations (|r| > 0.7):")
significant_corr = correlation_matrix[(correlation_matrix.abs() > 0.7) & (correlation_matrix != 1.0)].dropna(how='all', axis=0).dropna(how='all', axis=1)
print(significant_corr)
print("\n3. Profitability Analysis (Top 5 Firms):")
print(profitability_summary)
print("\n4. ANOVA Test Results:")
print(f"F-statistic: {anova_result.statistic:.2f}")
print(f"p-value: {anova_result.pvalue:.3f}")


import pandas as pd
import plotly.graph_objects as go


# Convert 'date' column to datetime format, handling potential mixed formats
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Handle rows where 'date' could not be converted
data = data.dropna(subset=['date'])

# Identify the top firm by Total_Revenue
top_firm = data.groupby('firm')['Total_Revenue'].sum().idxmax()
top_firm_data = data[data['firm'] == top_firm]

# Set the date column as the index
top_firm_data.set_index('date', inplace=True)

# Resample the data to monthly frequency, summing up the revenue and net income
monthly_aggregated = top_firm_data.resample('MS').agg({'Total_Revenue': 'sum', 'Net_Income': 'sum'}).fillna(0)

# Interactive Plotly chart
fig = go.Figure()

# Add Total Revenue time series
fig.add_trace(go.Scatter(x=monthly_aggregated.index, y=monthly_aggregated['Total_Revenue'], mode='lines+markers', name='Total Revenue'))

# Add Net Income time series
fig.add_trace(go.Scatter(x=monthly_aggregated.index, y=monthly_aggregated['Net_Income'], mode='lines+markers', name='Net Income'))

# Customize layout
fig.update_layout(
    title=f'Monthly Aggregated Financial Metrics for {top_firm} (2020)',
    xaxis_title='Date',
    yaxis_title='Amount',
    hovermode='x unified'
)

# Show the interactive plot
fig.show()

# Print the comprehensive report
print("\nComprehensive Financial Metrics Analysis Report")
print("-----------------------------------------------")
print(f"Top Firm: {top_firm}")
print(f"\n1. Monthly Aggregated Metrics:")
print(monthly_aggregated)