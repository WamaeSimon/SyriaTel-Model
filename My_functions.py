#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Create a function that imports the opens the data, prints the info, describe, shape
def dataframe(file_name):
    #read the data
    df = pd.read_csv(file_name)
    return df 


# In[ ]:


def data_info(dataframe):
    print("The dataframe has {} rows".format(dataframe.shape[0]))
    print("The dataframe has {} columns".format(dataframe.shape[1]))
          
    print('----------------------------------')
    print('Information:', dataframe.info())
          
    print('----------------------------------')
    print(dataframe.columns)
          
    print('----------------------------------')
    print('null_value:', dataframe.isnull().sum())

# Function that returns the numerical and categorical dataframes



# Function that returns the numerical and categorical dataframes/columsn
def Type_categorical(data_frame):
    categorical_df = data_frame.select_dtypes(include = 'object')
    return categorical_df
def Type_numerical (data_frame):
    numerical_df = data_frame.select_dtypes(include = 'number')
    return numerical_df  


    
def duplicates(data_frame):
    """function to check for the duplicates and return their value counts"""
    counts = data_frame.duplicated().value_counts()
    return counts


def outliers(data_frame):
    """ Function to check for outliers in each column and return their value counts"""
    col_names= []
    col_count = []
    for col in data_frame.columns:
        # Calculate the first and third quartiles (Q1 and Q3)
        Q1 = data_frame[col].quantile(0.25)
        Q3 = data_frame[col].quantile(0.75)
        # Calculate the interquartile range (IQR)
        IQR = Q3 -Q1
        # Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # find the value beyound the lower and upper bounds
        outliers = data_frame[col][(data_frame[col] < lower_bound) |
                           (data_frame[col] > upper_bound)]
        #append the column names and tye number of outliers
        col_names.append(col)
        col_count.append(outliers.value_counts().sum())
    return list(zip(col_names,col_count))


# Function to remove the outliers
def Remove_Outliers(dataframe):
    """
    A function that checks for the outliers beyond the bounds in each column and removes rows with those outliers
    then returns the new data frame without the outliers.
    """
    # Make a copy of the dataframe to avoid modifying the original dataframe
    df_cleaned = dataframe.copy()
    
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ['int64', 'float64']:
            # Calculate the first and third quartiles (Q1 and Q3)
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)

            # Calculate the interquartile range (IQR)
            IQR = Q3 - Q1

            # Define the lower and upper bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter out the rows with outliers in the current column
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        else: pass

    # Return the shape of the cleaned dataframe and the dataframe itself
    return df_cleaned




# create a function that takes in the categorical features/column and plots its distribution
def category_distributions(df, column):
    plt.figure(figsize=(12,6)) 
    sns.countplot(data=df, x=column, order = df[column].value_counts().index)  
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=90)
    plt.xlabel(column)
    plt.ylabel('Count')
    

    
### Create a function that plots the distribution of all the numerical features.
def Feature_distributions(df, bins=20, kde=True):
    """
    Plots the distribution of all features in the df.
    Parameters: we use the default parameters for number of bins,and kde
    """
    # Select only numerical columns from the DataFrame
    columns = df.columns
    
    # Number of subplots needed
    n = len(columns)
    
    # Determine number of rows and columns for the subplot grid
    # Number of columns in the plot grid
    n_cols = 3 
    # Calculate rows needed
    n_rows = (n // n_cols) + (1 if n % n_cols != 0 else 0)  

    # Create subplots
    plt.figure(figsize=(n_cols * 6, n_rows * 4)) 
    
    # Iterate over the numeric columns and plot the distributions
    for i, column in enumerate(columns):
        # Subplot index (1-based)
        plt.subplot(n_rows, n_cols, i + 1)  
        
        # Plot histogram with KDE
        sns.histplot(df[column], bins=bins, kde=kde)
        
        # Set the title and labels
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

    
### Functions for Bivariate Analysis
def categorical_bivariates(data, column):
    """ plotting the distribution of churn in each categorical feature"""
    counts = data.groupby(column)['churn'].sum().sort_values(ascending = False)
    plt.figure(figsize= (20,6))
    sns.countplot(x = column, data= data, hue = 'churn', order = counts.index )
    plt.xticks(rotation=90)
    plt.legend(loc="upper right")
    plt.show()
        
    
    
    
### Define a function that plots the kernel densities of the numerical features    
def plot_churn_kde(data):
    """
    A function to plot features based on churn rate for all the numerical features for easier comparison 
    """
    # Select only numerical columns from the DataFrame
    
    numerical_cols = data.select_dtypes(include=['number']).columns
    
    # Number of subplots needed
    n = len(numerical_cols)
    
    # Determine number of rows and columns for the subplot grid
    n_cols = 3 
    n_rows = (n // n_cols) + (1 if n % n_cols != 0 else 0)
    
    # Create subplots 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = axes.flatten()  # Flatten to iterate easily
    
    # Create a for loop to iterate over the columns and plot
    for i, col in enumerate(numerical_cols):
        sns.kdeplot(data=data, x=col,  hue ='churn', fill=True, ax=axes[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Churn Distribution by {col}')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()
    
    
    
# create a function to display the heatmap for the correlation matrix
def corr_matrix(data):
    """
        A function that calculates the correlation matrix and plots a heat map.
    """
        # perform the correlation on the data
    correlation_matrix = data.corr()
        #initialize the plot figure size
    plt.figure(figsize=(12, 10))
        # plot a heatmap for the correlationmatrix values, pass in annot= True meaning annotation ofthe matrices
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

        
