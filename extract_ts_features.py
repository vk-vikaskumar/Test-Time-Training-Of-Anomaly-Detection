import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import scipy.stats as stats
from scipy.stats import skew, kurtosis
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#from sklearn.preprocessing import StandardScaler

import gc

from scipy.ndimage import convolve
#scaler = StandardScaler()
#pca = PCA(n_components=1)  

#tsne = TSNE(n_components=1, random_state=42, perplexity=30, n_iter=1000)


def add_convolution_feature(df_col, kernel_size=3):
    """
    Add convolutional features to the DataFrame columns using a simple moving average kernel.

    Parameters:
    df (pd.DataFrame): Input DataFrame with time series data.
    kernel_size (int): Size of the convolution kernel (window size).

    Returns:
    pd.DataFrame: DataFrame with added convolutional features.
    """
    # Create a copy of the original DataFrame

    # Define a simple moving average kernel
    kernel = np.ones(kernel_size) / kernel_size

    # Apply convolution to each column
    convolved = convolve(df_col, kernel, mode='reflect')
    #df_convolved[f'{col}_convolved_kernel_{kernel_size}'] = convolved

    return convolved

def apply_box_cox_with_shift(series):
    """
    Apply Box-Cox transformation to a series with potential negative values.

    Parameters:
    series (pd.Series): Input time series data.

    Returns:
    pd.Series: Transformed series with Box-Cox applied.
    float: Lambda value used in Box-Cox transformation.
    float: Shift applied to make all values positive.
    """
    # Find the minimum value in the series
    min_value = series.min()

    # If there are negative values, shift the series to make all values positive
    shift = 0
    if min_value <= 0:
        shift = abs(min_value) + 1
        series = series + shift

    # Apply the Box-Cox transformation
    transformed_data, fitted_lambda = stats.boxcox(series)

    # Return the transformed data, lambda, and shift used
    return pd.Series(transformed_data, index=series.index), shift

def detect_peaks_and_features(time_series, distance=None, height=None):
    """
    Detect peaks in the time series and extract features related to peaks.
    """
    # Step 1: Detect peaks using scipy's find_peaks function
    peaks, properties = find_peaks(time_series, distance=distance, height=height)

    # Step 2: Extract peak features
    peak_heights = properties['peak_heights']  # Heights of the peaks
    peak_distances = np.diff(peaks)            # Distances between consecutive peaks
    temp_peak_value =  np.zeros(len(time_series))

    for i in range(len(peaks)):
        if i+1==len(peaks):
            #print(i, len(peaks))
            temp_peak_value[peaks[i-1]:] = peak_heights[i]

        elif i==0:
            temp_peak_value[:peaks[i]] = peak_heights[i]
        else:
            temp_peak_value[peaks[i-1]:peaks[i]] = peak_heights[i]

    return temp_peak_value


def apply_differencing(df, col, order=1):
    """
    Apply differencing iteratively to each column in the DataFrame up to the specified order.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing time series data.
    order (int): The maximum order of differencing to apply.

    Returns:
    pd.DataFrame: DataFrame with differenced time series data for each order.
    """

    df_diff = pd.DataFrame([])

    # Apply differencing iteratively for each order
    for current_order in range(1, order+1):
        # Compute the differencing
        diff = df.diff(periods=current_order).add_suffix(f'_diff_order_{current_order+1}')
        #print("diff:",diff.head(5))

        # Rename the columns to indicate the order of differencing
        diff_columns = f'{col}_diff_order_{current_order+1}'
        temp = pd.Series( data = diff.values, name=diff_columns)
        # Concatenate the original DataFrame with the new differenced columns
        #print("diff_col:", diff.columns)
        #print("diff:",diff.head(5))
        df_diff = pd.concat([df_diff, temp], axis=1)
        #print("df_diff_column",df_diff.columns)
        # Update df to the differenced data for the next iteration

        #print("done")
    return df_diff

def calculate_shannon_entropy(series):
    value_counts = series.value_counts(normalize=True, bins=10)  # Discretize data into 10 bins
    return entropy(value_counts)

def apply_time_series_transformations(df, test="no", window_size=5, wavelet='db1'):
    """
    Apply sliding window with rolling median, DFT, and DWT transformations to each column in the DataFrame.
    """
    df1 = df.copy()
    
    if "is_anomaly" in df.columns:
        df1 = df.iloc[:, :-1].copy()
    
    if test == "yes":
        df_transformed = df1.copy()

    # List to store the generated feature DataFrames
    feature_list = []

    for col in df1.columns:
        # Dictionary to hold the new columns for the current `col`
        col_features = {}
        
       
        col_features[f'{col}_train_median_current_value'] = df[col]-df[col].median()
        col_features[f'{col}_train_mean_current_value'] = df[col]-df[col].mean()
        #col_features[f'{col}_train_var_current_value'] = df[col].rolling(window=window_size).var()-df[col].var()



        col_features[f'{col}_train_median_current_value'] = df[col]-df[col].rolling(window=window_size).median()
        col_features[f'{col}_train_mean_current_value'] = df[col]-df[col].rolling(window=window_size).mean()

        # Feature 2: Peak Detection
        col_features[f'{col}_upper_peak_values'] = detect_peaks_and_features(df[col], distance=20, height=0.5)
        col_features[f'{col}_lower_peak_values'] = detect_peaks_and_features(df[col] * (-1), distance=20, height=0.5) * (-1)
        col_features[f'{col}_diff_upper_lower_peak_values'] = (
            col_features[f'{col}_upper_peak_values'] + col_features[f'{col}_lower_peak_values']
        )
        
        # Feature 3: DFT (Discrete Fourier Transform)
        dft_result = np.fft.fft(df1[col])
        col_features[f'{col}_dft_magnitude'] = np.abs(dft_result)

        # Step 4: Differencing (with order)
        temp_df = apply_differencing(df1[col], col, order=5)
        temp_df.columns = [f"{col}_{colname}" for colname in temp_df.columns]  # Adjusting the column names
        
        #step 5:
        diff_rr_1 = df[col].diff(periods=1)
        diff_rr_2 = df[col].diff(periods=2)
        col_features[f'{col}_RMSSD_1'] = np.sqrt(np.mean(diff_rr_2**2)) - df[col]
        col_features[f'{col}_RMSSD_2'] = np.sqrt(np.mean(diff_rr_2**2)) - df[col]
        #Step 6 pNN50 (percentage of successive RR intervals > 50ms)

        #pnn50_count = np.sum(np.abs(diff_rr_1) > 50)
        pnn100_count = np.sum(np.abs(diff_rr_2) > 100)
        col_features[f'{col}_pnn50']= (pnn100_count / len(diff_rr_2)) * 100 - df[col]

        # Convert Degrees to Radians
        angular_data_radians = np.radians(df[col])

        # Step 2: Compute Sine and Cosine
        col_features[f'{col}_sine_comp'] = np.sin(angular_data_radians)
        col_features[f'{col}_cos_comp'] = np.cos(angular_data_radians)

        #col_features[f'{col}_pca'] = pca.fit_transform(df[col])
        #col_features[f'{col}_tsne'] = tsne.fit_transform(df[col])

        #step 7: convolution
        #col_features[f'{col}_convolution'] = add_convolution_feature(df[col], kernel_size=5)
        # Append this column's features to the feature list
        feature_list.append(pd.DataFrame(col_features))
        feature_list.append(temp_df)  # Add differencing DataFrame
    del col_features
    gc.collect()
    # Combine all the features into a single DataFrame using concat
    df_transformed = pd.concat([df1] + feature_list, axis=1)

    # Fill NaNs with the median
    df_transformed = df_transformed.fillna(df_transformed.median())

    # Optionally, append original dataset back if required (if test == 'no')
    df_transformed = pd.concat([df_transformed, df], axis=1)

    return df_transformed
