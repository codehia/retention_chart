from operator import attrgetter

import matplotlib.colors as mcolors
# data visualization libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class RetentionChart:
    def __init__(self, file_path='data/retention.json'):
        self.data = pd.read_json(file_path)
        self.format_data()
        self.clean_data()
        self.create_cohort_labels()
        self.retention_matrix, self.cohort_size = self.create_retention_matrix()
        self.create_chart(self.retention_matrix, self.cohort_size)

    def format_data(self):
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

    def clean_data(self):
        self.data.drop(self.data[self.data['timestamp'].dt.year == 1970].index, inplace=True)

    def create_cohort_labels(self):
        # Convert the 'timestamp' column to a period representation in terms of months
        self.data['inference_month'] = self.data['timestamp'].dt.to_period('M')
        # Group the data by 'user_id', find the minimum 'timestamp' for each group,
        # convert it to a period representation in terms of months, and assign it to a new column 'cohort'
        self.data['cohort'] = self.data.groupby('user_id')['timestamp'].transform('min').dt.to_period('M')

    def create_retention_matrix(self):
        # Group the data by 'cohort' and 'inference_month', count unique 'user_id's for each group, and reset the index
        df_cohort = self.data.groupby(['cohort', 'inference_month']).agg(
            users_count=('user_id', 'nunique')).reset_index(drop=False)
        # Calculate the period number by subtracting the 'cohort' from 'inference_month'
        df_cohort['period_number'] = (df_cohort.inference_month - df_cohort.cohort).apply(attrgetter('n'))
        # Pivot the DataFrame to create a cohort analysis table
        cohort_pivot = df_cohort.pivot_table(index='cohort', columns='period_number', values='users_count')
        # Get the size of each cohort
        cohort_size = cohort_pivot.iloc[:, 0]
        # Return the cohort analysis table and the size of each cohort
        return cohort_pivot.divide(cohort_size, axis=0), cohort_size

    def create_chart(self, retention_matrix, cohort_size):
        # Set the aesthetic style of the plots
        with sns.axes_style("white"):
            # Create a 1x2 subplot with shared y axis and specific width ratios
            fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
            # Plot the retention matrix as a heatmap
            sns.heatmap(retention_matrix, mask=retention_matrix.isnull(), annot=True, fmt='.0%', cmap='RdYlGn', ax=ax[1])  # noqa
            # Set the title and labels of the heatmap
            ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
            ax[1].set(xlabel='# of periods', ylabel='')
            # Create a DataFrame for cohort size and rename the column
            cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
            # Create a white colormap for the heatmap
            white_cmap = mcolors.ListedColormap(['white'])
            # Plot the cohort size as a heatmap
            sns.heatmap(cohort_size_df, annot=True, cbar=False, fmt='g', cmap=white_cmap, ax=ax[0])
            # Adjust the layout of the plot
            fig.tight_layout()
            # Save the plot as a PNG file
            plt.savefig('output/retention.png')


if __name__ == '__main__':
    retention = RetentionChart()
