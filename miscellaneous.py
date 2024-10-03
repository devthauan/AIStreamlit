import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import json
# from .rag import RagFunctions
from sklearn.preprocessing import StandardScaler

def apply_embedding_value_correction(x : list) -> list:
    """
    Applies a correction to the values in the input list of embeddings.

    This function processes a list of embeddings, where each embedding can be either a NumPy array, a list, or a 
    NumPy generic type. The function converts each embedding to a NumPy array of floats and ensures that all values 
    within the embedding are of type float. If an embedding is of an unknown type or contains values that are not floats, 
    it raises a TypeError.

    Parameters:
    - x (list): A list of embeddings, where each embedding can be a NumPy array, a list, or a NumPy generic type.

    Returns:
    - corrected_values (list): A list of corrected embeddings where each embedding is a NumPy array of floats.

    Raises:
        TypeError: If an element in the input list is neither a NumPy array nor a list, or if any value in the embeddings is not a float.
    """
    import numpy as np
    corrected_values = []
    for embedding in x:
        if isinstance(embedding, (np.ndarray, np.generic) ):
            corrected_values.append(embedding.astype(float))
        elif isinstance(embedding,list):
            corrected_values.append(np.array(embedding).astype(float))
        else:
            print("Unknow type ",type(embedding))
            raise TypeError("Unknow type ",type(embedding))
        for value in embedding:
            if not isinstance(value,float):
                print(f'The value {value} is not a float.')
                raise TypeError(f'The value {value} is not a float.')
    return corrected_values

def compute_metrics(x : list ,kmeans_result : dict) -> dict:
    """
    Computes clustering evaluation metrics for different KMeans clustering results.

    This function calculates several clustering evaluation metrics, including inertia, silhouette score, 
    Davies-Bouldin score, and Calinski-Harabasz score for each set of KMeans clustering results provided.

    Parameters:
    - x (list): The data used for clustering, where each element in the list represents a data point 
                  (e.g., a list or array-like structure) and each data point is a feature vector.
    - kmeans_result (dict): A dictionary where keys are the number of clusters and values are the corresponding
                              KMeans clustering results (sklearn.cluster.KMeans instances). Each KMeans instance 
                              should be fitted on the data 'x'.

    Returns:
    - metrics_dict (dict): A dictionary with lists of metric values for each number of clusters. The keys are:
            - 'inertia': List of inertia values for each KMeans result.
            - 'silhouette': List of silhouette scores for each KMeans result.
            - 'davies_bouldin': List of Davies-Bouldin scores for each KMeans result.
            - 'calinski_harabasz': List of Calinski-Harabasz scores for each KMeans result.

    Raises:
        ValueError: If 'x' is not a list or if elements in 'x' are not compatible with the KMeans results.
        ValueError: If 'kmeans_result' does not contain valid KMeans results or if the data 'x' is not compatible
                    with the KMeans results.
    """
    from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score

    inertia_result,silhouette_score_result,davies_bouldin_score_result,calinski_harabasz_result = [],[],[],[]

    for n_clusters,kmeans in kmeans_result.items():
        inertia_result.append(kmeans.inertia_)
        silhouette_score_result.append(silhouette_score(x, kmeans.labels_))
        davies_bouldin_score_result.append(davies_bouldin_score(x, kmeans.labels_))
        calinski_harabasz_result.append(calinski_harabasz_score(x, kmeans.labels_))

    metrics_dict = {
        'inertia': inertia_result,
        'silhouette': silhouette_score_result,
        'davies_bouldin': davies_bouldin_score_result,
        'calinski_harabasz': calinski_harabasz_result
        }
    return metrics_dict

def compute_silhouette_score(x,kmeans_result):
    """
    Plots silhouette analysis for KMeans clustering results.

    This function generates silhouette plots to evaluate the quality of clustering results for different numbers of clusters 
    using KMeans clustering. The silhouette plot provides insight into how well-separated and how well-clustered the 
    clusters are by visualizing the silhouette coefficient for each sample.

    Parameters:
    - x (list): The data used for clustering, where each element in the list represents a data point 
                  (e.g., a list or array-like structure) and each data point is a feature vector.
    - kmeans_result (dict): A dictionary where keys are the number of clusters and values are the corresponding
                              KMeans clustering results (sklearn.cluster.KMeans instances). Each KMeans instance 
                              should be fitted on the data 'x'.

    Returns:
        None: This function generates and displays silhouette plots but does not return any value.

    Raises:
        ValueError: If 'x' is not a list or if elements in 'x' are not compatible with the KMeans results.
        ValueError: If 'kmeans_result' does not contain valid KMeans results or if the data 'x' is not compatible
                    with the KMeans results.
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    for n_clusters,kmeans in kmeans_result.items():
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(16, 9)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

        cluster_labels = kmeans.labels_#.predict(x)
        silhouette_avg = silhouette_score(x, cluster_labels)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(x, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

        plt.show()

def build_hypterparameter_loop(
        hyperparameter_dict : dict
        ):
    """
    This function generates all possible combinations of hyperparameters from a dictionary of hyperparameter 
    values and returns them as a list of dictionaries. 

    Parameters:
    hyperparameter_dict (dict): 
        A dictionary where the keys represent the names of hyperparameters, 
        and the values are lists containing the possible values for those hyperparameters.

    Returns:
        The list of dictionaries, where each dictionary represents one possible combination of hyperparameters.
    """
    from itertools import product

    # Extract parameter names and their corresponding lists of values
    parameter_names = list(hyperparameter_dict.keys())
    values_lists = [values for values in hyperparameter_dict.values()]

    # Generate all combinations
    all_combinations = list(product(*values_lists))

    # Create dictionaries for each combination
    combination_dicts = []
    for combination in all_combinations:
        combination_dict = {parameter_names[i]: combination[i] for i in range(len(parameter_names))}
        combination_dicts.append(combination_dict)

    return combination_dicts

def valid_hallucination_yes_no(
    data : pd.DataFrame,
    column_1 : str,
    column_2 : str = '',
    categoric : bool = True
    ):

    """
    This function creates a side-by-side bar chart representing the percentage distribution 
    of two categorical variables from a dataset. The chart shows the percentages of occurrences 
    in column_2 (e.g., "Yes" and "No") for each unique value in column_1.

    Parameters:
    data (pd.DataFrame): 
        The DataFrame containing the dataset to be visualized.
    column_1 (str): 
        The first categorical column to be used as the grouping variable (x-axis).
    column_2 (str): 
        The second categorical column, with values to be represented as percentage bars (e.g., 'Yes' and 'No').
    categoric (bool):
        Parameter to determine which plot will be returned, depending on categorization or not.

    Returns:
        Bar graphic.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if categoric == True:
        counts = data.groupby([column_1, column_2]).size().unstack(fill_value=0)

        # Percentage calculation
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        
        # Graphic configuration
        bar_width = 0.35
        index = np.arange(len(percentages))

        # Side by sile bar creation
        fig, ax = plt.subplots()
        bar1 = ax.bar(index, percentages['Yes'], bar_width, label='Yes', color =  'skyblue')
        bar2 = ax.bar(index + bar_width, percentages['No'], bar_width, label='No', color =  'lightgreen')

        # Add values ​​to bars
        for rect in bar1 + bar2:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.1f}%', ha='center', va='bottom')

        # Graphic configuration
        ax.set_xlabel(column_1)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{column_2} by {column_1}')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(percentages.index)
        ax.legend()

        plt.show()

    else:

        percentages = data[column_1].value_counts(normalize = True).sort_index(ascending=False)
        percentages = pd.DataFrame(percentages)

        # Percentage calculation
        percentages['proportion'] = percentages['proportion'] * 100
        # Graphic configuration
        bar_width = 0.20
        index = np.arange(len(percentages))

        # Side by sile bar creation
        fig, ax = plt.subplots()
        bar1 = ax.bar(index, percentages['proportion'])

        # Add values ​​to bars
        for rect in bar1:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.1f}%', ha='center', va='bottom')

        # Graphic configuration
        ax.set_xlabel(column_1)
        ax.set_ylabel('Percentage (%)')
        ax.set_xticks(index)
        ax.set_xticklabels(percentages.index)
        ax.legend()

        plt.show()

def valid_hallucination_1_to_5(
        data : pd.DataFrame,
        column : str,
        title : str = f'Hallucination distribution',
        category : bool = True
):
    """
    This function creates a side-by-side bar chart that shows the percentage distribution 
    of hallucination ratings (1 to 5) for different categories in a specified column. 

    Parameters:
    data (pd.DataFrame): 
        The DataFrame containing the dataset, including the column with categories and a 'Value' column that likely contains hallucination ratings (1 to 5).
    column (str): 
        The column name that represents the categories for which the hallucination ratings will be compared (e.g., 'Template_name').
    title (str):
        Graphic title.

    Returns:
        Bar graphic.
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if category == True:
        counts = data.groupby([column, 'Value']).size().unstack(fill_value=0)
        counts = counts.reset_index()

        # Value categories
        categories = [1, 2, 3, 4, 5]

        # Convert to percentage
        counts[categories] = counts[categories].div(counts[categories].sum(axis=1), axis=0) * 100

        # Set the width of the bars and the number of determinated column
        width = 0.15
        x = np.arange(len(categories))

        fig, ax = plt.subplots(figsize = (21,9))

        # Colors definement
        if column == 'Template_name':
            colors = {'detailed_response': 'skyblue', 'short_response': 'lightgreen'}

            # Plot bars side by side for each parameter
            for idx, row in counts.iterrows():
                ax.bar(x + idx * width, row[categories], width=width, label=f'{column}: {row[column]}', color=colors[row[column]])

        else:
            colors = plt.cm.winter(counts[column] / counts[column].max())

            # Plot bars side by side for each parameter
            for idx, row in counts.iterrows():
                ax.bar(x + idx * width, row[categories], width=width, label=f'{column}: {row[column]}', color=colors[idx])


        # Add values ​​to bars
        for idx, row in counts.iterrows():
            for i, cat in enumerate(categories):
                ax.text(x[i] + idx * width, row[cat] + 1, f'{row[cat]:.1f}%', ha='center', va='bottom')

        ax.set_xlabel('Hallucination 1 to 5')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(counts) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend(title=column)

        plt.show()

    else:

        counts = data.groupby([column, 'Hallucination 1 to 5']).size().unstack(fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        percentages = percentages.T

        bar_width = 0.25
        index = np.arange(len(percentages))

        fig, ax = plt.subplots(figsize = (8, 6))

        colors = sns.color_palette(['lightgreen', 'lightblue', 'dodgerblue',])

        bars = []
        for i, category in enumerate(percentages.columns):
            bars.append(ax.bar(index + i * bar_width, percentages[category], bar_width, label=category, color=colors[i % len(colors)]))

        for bar in bars:
            for rect in bar:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.1f}%', ha='center', va='bottom')

        ax.set_xlabel('Hallucination 1 to 5')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'Hallucination 1 to 5 by {column}')
        ax.set_xticks(index + bar_width * 1.5)
        ax.set_xticklabels(percentages.index)
        ax.legend()

        plt.tight_layout()
        plt.show()

def pivot_table(
        data : pd.DataFrame,
        column_1 : str,
        column_2 : str,
        title : str = '',
):
    
    """
    This function generates a stacked bar chart based on a cross-tabulation of two columns from a dataset. 
    It counts the frequency of occurrences in one column (`column_1`) for each category in another column (`column_2`).

    Parameters:
    data (pd.DataFrame): 
        The DataFrame containing the dataset.
    column_1 (str): 
        The column whose values will be represented on the x-axis of the bar chart.
    column_2 (str): 
        The column whose values will be used to segment the bars (stacked on top of each other).
    title (str, optional): 
        The title for the bar chart. If not provided, it defaults to an empty string.

    Returns:
        The function returns the cross-tabulated DataFrame, 
        which can be further analyzed or used in other computations.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    # Create a cross-tab frequency table to count the values ​​of column_1 for each column_1
    cross_tab = pd.crosstab(data[column_1], data[column_2])

    ax = cross_tab.plot(kind='bar', stacked=True, figsize=(12, 8))

    limite_y = 30  # Only display values ​​on bars with height greater than 30

    # Add the values ​​in the bars that exceed the limit
    for container in ax.containers:
        labels = [f'{int(height)}' if height > limite_y else '' for height in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center')

    
    plt.title(title)
    plt.xlabel(column_1)
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    return cross_tab

def plot_2D_or_3D(
        data : pd.DataFrame,
        x : str,
        y : str,
        z : str = '',
        scale : str = 'No',
        dimention : str = '2D'
):
    """
    This function generates either a 2D or 3D scatter plot based on the input data. 
    It plots two or three columns of a DataFrame on the specified axes and can scale the color of the points according to a specified column.

    Parameters:
    data (pd.DataFrame): 
        The input dataset containing the columns to be visualized.
    x (str): 
        The name of the column in the DataFrame to be used for the X-axis.
    y (str): 
        The name of the column in the DataFrame to be used for the Y-axis.

    z (str, optional): 
        The name of the column in the DataFrame to be used for the Z-axis in 3D plots. If empty, the function defaults to a 2D plot. Default is an empty string ('').

    scale (str, optional): 
        The name of the column used to scale the color of the points in the plot. If set to 'No', no color scaling is applied. Default is 'No'.

    dimention (str, optional): 
        Specifies whether to generate a 2D or 3D plot. Accepts '2D' or '3D'. Default is '2D'.

    Returns
    3D Plot: 
        If dimention is set to '3D', the function creates a 3D scatter plot where x, y, and z are mapped to the X, Y, and Z axes, respectively. 

    2D Plot: 
        If dimention is set to '2D', the function creates a 2D scatter plot with x on the X-axis and y on the Y-axis. 
    """
    import matplotlib.pyplot as plt

    if dimention == '3D':
    
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
    
        scatter = ax.scatter(data[x], data[y], data[z], c=data[scale], cmap='GnBu', s=100)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title('No Hallucination Percentage by Parameters')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(scale)

        plt.show()
    
    else:

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data[x], data[y], c=data[scale], cmap='GnBu', s=100)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'No Hallucination Percentage by {x} and {y} with Top_p=1')

        cbar = plt.colorbar(scatter)
        cbar.set_label(scale)

        plt.show()

def calculate_outliers_with_limits(data : pd.DataFrame):

    """
    This function calculates the lower and upper limits for detecting outliers in each 
    column of a DataFrame based on the Interquartile Range (IQR) method. 
    The limits are returned in the form of a dictionary.

    Parameters
    data (pd.DataFrame): 
        The DataFrame containing the data for which outlier limits are to be calculated. Each column should contain numerical values.

    Returns
        The limits_dict dictionary, which contains the calculated limits for detecting outliers in each column.
    """

    # Check for duplicate columns and issue a warning if any are found
    if data.columns.duplicated().any():
        
        duplicated_cols = data.columns[data.columns.duplicated()].tolist()
        warnings.warn(f"Warning: The dataset contains duplicate columns: {duplicated_cols}")

    limits_dict = {}
    
    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
                
        # Store limits in dictionary
        limits_dict[col] = {
            'lower_limit': lim_inf, 
            'upper_limit': lim_sup
            }
    
    return limits_dict

def remove_outliers(
        data : pd.DataFrame, 
        limits_dict : dict
        ):

    """
    This function removes rows from a DataFrame where the values in each column fall 
    outside the specified lower and upper limits. 
    These limits are typically calculated using the Interquartile Range (IQR) 
    or another method and are passed as a dictionary.

    Parameters
    data (pd.DataFrame): 
        The DataFrame from which outliers need to be removed. The DataFrame contains numerical data in each column.
    limits_dict (dict): 
        A dictionary containing the lower and upper limits for each column in the DataFrame.

    Returns
        The modified DataFrame, with rows containing outliers removed.
    """
    
    for col, limits in limits_dict.items():

        lower_limit = limits['lower_limit']
        upper_limit = limits['upper_limit']
        
        data[col] = np.where((((data[col] < lower_limit) | ((data[col] > upper_limit)))), (np.nan), (data[col]))
    
    return data

def apply_rolling_mean(
        data : pd.DataFrame, 
        variable_name : str, 
        window : str
                ):
    
    """
    Function responsible for applying rolling mean to a dataframe column.
    """
    
    return data[variable_name].rolling(window).mean()

def get_rolling_mean(
        data : pd.DataFrame, 
        list_column : list,
        list_windows : list = ['1min','5min','10min','20min','30min','45min','1H', '1.5H','2H',]
        ):
    
    """
    Calculates rolling means for specified columns and time windows, and computes the difference 
    between the original data and the rolling mean.

    Parameters
    -----------
    data (pd.DataFrame): 
        The DataFrame containing the data on which rolling means will be applied.
    list_column (list): 
        A list of column names for which rolling means will be calculated.
    list_windows (list): 
        A list of time window sizes for calculating the rolling mean (default includes a 
        variety of minute and hour windows).

    Returns
    --------
    pd.DataFrame
        The original DataFrame with additional columns for the rolling mean and 
        the difference between the original values and the rolling mean.
    """


    for column in list_column:
                
        for window in list_windows:

            data[f'{column}_{window}'] = apply_rolling_mean(data, variable_name = column, window = window)
            data[f'dif_{column}_{window}'] = data[f'{column}'] - data[f'{column}_{window}']

    return data

def fillna_with_interpolation(
        data : pd.DataFrame, 
        column : str):

    """
    This function fills missing values (NaN) in a specific column of a DataFrame using linear interpolation.

    Parameters
    data (pd.DataFrame): 
        The DataFrame containing the data.
    column (str): 
        The name of the column in which missing values (NaN) should be filled using interpolation.

    Returns
        The updated DataFrame is returned with the specified column's NaN values filled.
    """
    data[column] = data[column].interpolate(method='linear')

    return data

def fillna_with_last_value(
        data : pd.DataFrame, 
        column : str
        ):

    """
    Fills missing (NaN) values in the specified column of the 
    DataFrame using the last valid value (forward fill method).

    Parameters
    -----------
    data (pd.DataFrame): 
        The input DataFrame containing the data with missing values.
    column (str):
        The name of the column in which NaN values will be filled with the last observed value.

    Returns
    --------
    pd.DataFrame
        The DataFrame with NaN values in the specified column replaced by the last valid value.
    """

    data[column] = data[column].fillna(method = 'ffill')

    return data

def merge_to_streamlit(
        data_inputs : pd.DataFrame,
        data_outputs : pd.DataFrame,
        path : str,
        colum_output : str
        ):

    """
    Merges input and output datasets on the 'timestamp' column, preparing the 
    data for display in a Streamlit application.

    Parameters
    -----------
    data_inputs (pd.DataFrame): 
        The input DataFrame containing the input variables for the 
        model or analysis. It is filtered to include only the columns 
        specified in `columns_input`.
    data_outputs (pd.DataFrame): 
        The output DataFrame containing the output data 
        (e.g., predictions or anomaly scores). It is filtered to include only 
        the column specified in `colum_output`.
    path (str): 
        File path with inputs columns name.
    colum_output (str): 
        The name of the output column in `data_outputs` that contains the 
        output or anomaly probability to be merged.

    Returns
    --------
    pd.DataFrame
        A DataFrame with the input data and the corresponding output (anomaly probability), 
        aligned by the 'timestamp' column.
    """

    with open(path, 'r') as file:
        input_tags = json.load(file)
    
    columns_input = [tag['Name'] for tag in input_tags['InputTags']]
    rename_input = [tag['Display'] for tag in input_tags['InputTags']]

    data_inputs = data_inputs[columns_input]

    data_outputs = data_outputs[[colum_output]]

    for old_name, new_name in zip(columns_input, rename_input):

        data_inputs.rename(columns={old_name: new_name}, inplace = True)

    data_outputs.rename(columns={colum_output: 'anomaly_probability'}, inplace = True)

    df = pd.concat([data_inputs, data_outputs], axis = 1)

    return df