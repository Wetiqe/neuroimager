import pandas as pd


# explore a give dataframe. These are scripts from an author on kaggle which I forgot the name.
# TODO: I will add the name once I find it.


def Common_data_analysis(
    data,
    missing_value_highlight_threshold=10.0,
    display_df=True,
    only_show_missing=False,
):
    """
    Analsis the common data type and give a table of info such as MMM (mean median mode) ,sd,var etc.

    :param data: dataframe.
    :param missing_value_highlight_threshold: Float the column having missing value percentage above this will be highlighted.

    return: list of all columns and list of categorical column and list of numerical column.
    """
    print("{:=^100}".format(" Common data analysis "))
    # default settings.
    column = data.columns
    total_samples = data.shape[0]
    value_dict = {}

    # calculate values.
    missing_values = data.isnull().sum().values
    missing_value_percentage = [
        round((col_missing_count / total_samples) * 100, 2)
        for col_missing_count in missing_values
    ]
    datatype = [data.iloc[:, i].dtype for i in range(data.shape[1])]

    categorical_data = list(data.loc[:, data.dtypes == "object"].columns)
    numerical_data = [d for d in column if d not in categorical_data]
    # print the diff datatype and count.
    print()
    print(
        "Numerical data list {} ---> total {} numerical values".format(
            numerical_data, len(numerical_data)
        )
    )
    print(
        "Categorical data list {} ---> total {} categorical values".format(
            categorical_data, len(categorical_data)
        )
    )
    print()

    # find is there any special character that represents the missing value.
    missing_value_by_character = [
        data.loc[data[col].isin(["?", " ", "/", "[", "]", "na", "null"]), col]
        .value_counts()
        .sum()
        for col in column
    ]
    missing_value_percentage_by_char = [
        round(col_missing_count / total_samples * 100, 2)
        for col_missing_count in missing_value_by_character
    ]
    # organise values.
    value_dict["data type"] = datatype
    value_dict["Missing Value(NA)"] = missing_values
    value_dict["?[]na null ' ' "] = missing_value_by_character
    value_dict["% of Missing value(NA)"] = missing_value_percentage
    value_dict["% of Missing value(?[]na null ' ')"] = missing_value_percentage_by_char
    df = pd.DataFrame(value_dict, columns=value_dict.keys(), index=column)

    # make a highlight for col has high missing value percentage. (>55% say)
    # the particular row will be highlighted if it above missing value threshold.
    def highlight_high_missing_value(sample):
        threshold = missing_value_highlight_threshold
        style = sample.copy()
        highlight = "background-color: red;"
        if sample[3] > threshold or sample[4] > threshold:
            style[:] = highlight
        else:
            style[:] = ""
        return style

    df_styled = df.style.apply(highlight_high_missing_value, axis=1)
    if display_df:
        if only_show_missing:
            df = df[df["Missing Value(NA)"] > 0]
            display(df)
        else:
            display(df_styled)
    return (column, categorical_data, numerical_data, df)


# columns, categorical_col, numerical_col  = Common_data_analysis(data, 5.0)
#
def numerical_data_analysis(data, numerical_data):
    """
    It will provide the analysis of the numerical data.

    :param data: dataframe.
    :param numerical_data: list of numerical column names.
    """
    print("{:=^100}".format(" Numerical data analysis "))
    print("The skewness we are calculatinf here has a value range ===")
    print("If skew is == 0 -----------> Perfect normal distribution(green color)")
    print("If skew is > +1 -----------> Highly positive skew(red color)")
    print("If skew is > -1 -----------> Highly negative skew(red color)")
    print("If  0 > skew < 0.5 --------> Moderate positive skew(blue color)")
    print("If  0 < skew > -0.5 -------> Moderate negative skew(blue color)")
    column = data.columns

    min_value = [data[col].min() if col in numerical_data else "NA" for col in column]
    max_value = [data[col].max() if col in numerical_data else "NA" for col in column]
    # mode_value = [data[col].mode() if col in numerical_data else "NA" for col in column]
    mean_value = [data[col].mean() if col in numerical_data else "NA" for col in column]
    std_value = [data[col].std() if col in numerical_data else "NA" for col in column]
    # print(mode_value)
    skewness_value = [
        data[col].skew() if col in numerical_data else "NA" for col in column
    ]
    kurtosis_value = [
        data[col].kurtosis() if col in numerical_data else "NA" for col in column
    ]

    q1_value = [
        data[col].quantile(0.25) if col in numerical_data else "NA" for col in column
    ]
    q2_meadian_value = [
        data[col].quantile(0.50) if col in numerical_data else "NA" for col in column
    ]
    q3_value = [
        data[col].quantile(0.75) if col in numerical_data else "NA" for col in column
    ]

    # find the range value.
    def find_range(min_value_list, max_value_list):
        range_value = [
            (max_value - min_value) if min_value != "NA" else "NA"
            for max_value, min_value in zip(max_value_list, min_value_list)
        ]
        return range_value

    # find the inter quartile range. (q3-q1)
    def iqr(q1_value_list, q3_value_list):
        range_value = [
            (q3 - q1) if q1 != "NA" else "NA"
            for q3, q1 in zip(q3_value_list, q1_value_list)
        ]
        return range_value

    range_value = find_range(min_value, max_value)
    iqr_value = iqr(q1_value, q3_value)

    # organise everything inside a dataframe.
    df_dict = {}
    df_dict["min"] = min_value
    df_dict["max"] = max_value
    df_dict["range(max-min)"] = range_value
    # df_dict["mode"] = mode_value
    df_dict["mean/average"] = mean_value
    df_dict["standard deviation"] = std_value
    df_dict["Q1"] = q1_value
    df_dict["meadian/Q2"] = q2_meadian_value
    df_dict["Q3"] = q3_value
    df_dict["Inter quantile range"] = iqr_value
    df_dict["kurtosis"] = kurtosis_value
    df_dict["Skewness"] = skewness_value

    df = pd.DataFrame(df_dict, columns=df_dict.keys(), index=column)

    # highlight the data based on its skewness.
    def highlight_skewness(sample):
        # make a style as the sample shape and property.
        style = sample.copy()
        # make other cell_value as empty style , because i am focusing in coloring skewness column only.
        style[:] = ""
        # set the colors for skewness cells.
        highly_skewed = "background-color: red;"
        moderatly_skewed = "background-color: blue;"
        perfect_normal_destribution = "background-color: green;"

        # color the cells
        if sample[-1] > 1 or sample[-1] < -1:
            style[-1] = highly_skewed
        elif (sample[-1] > 0.5 or sample[-1] <= 1) or (
            sample[-1] > -0.5 or sample[-1] <= -1
        ):
            style[-1] = moderatly_skewed
        elif sample[-1] == 0:
            style[:] = perfect_normal_destribution
        else:
            style[:] = ""
        return style

    df = df.style.apply(highlight_skewness, axis=1)
    display(df)


# Find how much outliers it has using the IQR methods(if data skewed) and z-score method(if data is normally distributed).
def find_outlier_z_score_method(
    data, new_feature=False, col_name=None, return_limits=False
):
    """
    Find the outliers in the given dataset.
    :param data: dataset has number of features to find the outliers.
    :param new_feature: If True create a new feature in the dataFrame to indicate this sample has any outlier feature else do nothing.
    :param return_limit: If true return the upper and lower limits.

    :return data with new feature and number of outlier in each features if new_feature is True, else only number of outlier in each features.
    """
    df = data.copy()
    mean_each_features = df.mean(axis=0)
    std_each_features = df.std(axis=0)
    lower_limit_each_feature = mean_each_features - (3 * std_each_features)
    upper_limit_each_feature = mean_each_features + (3 * std_each_features)

    # find the data is a outlier value or not.
    # print(lower_limit_each_feature, upper_limit_each_feature)
    outlier_df = (df > upper_limit_each_feature) | (df < lower_limit_each_feature)
    # find the number of outliers per feature.
    number_of_outlier_each_feature = outlier_df.sum(axis=0)
    if df.ndim == 1:
        # if the given data is features than handle diffrently. because it has no .values and .index function.
        number_of_outlier_each_feature_df = pd.DataFrame(
            {
                "Feature": [col_name if col_name else "Given feature"],
                "Number of outliers": number_of_outlier_each_feature,
            }
        )
    else:
        number_of_outlier_each_feature_df = pd.DataFrame(
            {
                "Features": number_of_outlier_each_feature.index,
                "Number of outliers": number_of_outlier_each_feature.values,
            }
        )

    # find the upper limit and lower limit Dataframe.
    if df.ndim == 1:
        # if the given data is features than handle diffrently. because it has no .values and .index function.
        lower_limit_each_feature_df = pd.DataFrame(
            {
                "Feature": [col_name if col_name else "Given feature"],
                "lower limit": lower_limit_each_feature,
            }
        )
        upper_limit_each_feature_df = pd.DataFrame(
            {
                "Feature": [col_name if col_name else "Given feature"],
                "upper limit": upper_limit_each_feature,
            }
        )
    else:
        lower_limit_each_feature_df = pd.DataFrame(
            {
                "Features": lower_limit_each_feature.index,
                "lower limit": lower_limit_each_feature.values,
            }
        )
        upper_limit_each_feature_df = pd.DataFrame(
            {
                "Features": upper_limit_each_feature.index,
                "upper limit": upper_limit_each_feature.values,
            }
        )

    if new_feature and return_limits:
        df["num_of_outliers"] = outlier_df.sum(axis=1)
        return (
            df,
            number_of_outlier_each_feature_df,
            lower_limit_each_feature_df,
            upper_limit_each_feature_df,
        )
    elif new_feature:
        # add the new feature indicating this row has outlier data to the data and return.
        df["num_of_outliers"] = outlier_df.sum(axis=1)
        return df, number_of_outlier_each_feature_df

    elif return_limits:
        return lower_limit_each_feature_df, upper_limit_each_feature_df

    return number_of_outlier_each_feature_df


def find_outliers_iqr_method(
    data, new_feature=False, col_name=None, return_limits=False
):
    """Find the outliers in the given dataset.
    :param data: dataset has number of features to find the outliers.
    :param new_feature: If True create a new feature in the dataFrame to indicate this sample has any outlier feature else do nothing.
    :param return_limit: If true return the upper and lower limits.

    :return data with new feature and number of outlier in each features if new_feature is True, else only number of outlier in each features.
    """
    df = data.copy()
    q1_each_features = df.quantile(
        0.25,
    )
    q3_each_features = df.quantile(
        0.75,
    )
    iqr_each_feature = q3_each_features - q1_each_features
    lower_limit_each_feature = q1_each_features - (iqr_each_feature * 1.5)
    upper_limit_each_feature = q3_each_features + (iqr_each_feature * 1.5)

    # find the data is a outlier value or not.
    # print(lower_limit_each_feature, upper_limit_each_feature)
    outlier_df = (df > upper_limit_each_feature) | (df < lower_limit_each_feature)

    # find the number of outliers per feature.
    number_of_outlier_each_feature = outlier_df.sum(axis=0)
    if df.ndim == 1:
        # if the given data is features than handle diffrently. because it has no .values and .index function.
        number_of_outlier_each_feature_df = pd.DataFrame(
            {
                "Feature": [col_name if col_name else "Given feature"],
                "Number of outliers": number_of_outlier_each_feature,
            }
        )
    else:
        number_of_outlier_each_feature_df = pd.DataFrame(
            {
                "Features": number_of_outlier_each_feature.index,
                "Number of outliers": number_of_outlier_each_feature.values,
            }
        )

    # find the upper limit and lower limit Dataframe.
    if df.ndim == 1:
        # if the given data is features than handle diffrently. because it has no .values and .index function.
        lower_limit_each_feature_df = pd.DataFrame(
            {
                "Feature": [col_name if col_name else "Given feature"],
                "lower limit": lower_limit_each_feature,
            }
        )
        upper_limit_each_feature_df = pd.DataFrame(
            {
                "Feature": [col_name if col_name else "Given feature"],
                "upper limit": upper_limit_each_feature,
            }
        )
    else:
        lower_limit_each_feature_df = pd.DataFrame(
            {
                "Features": lower_limit_each_feature.index,
                "lower limit": lower_limit_each_feature.values,
            }
        )
        upper_limit_each_feature_df = pd.DataFrame(
            {
                "Features": upper_limit_each_feature.index,
                "upper limit": upper_limit_each_feature.values,
            }
        )

    if new_feature and return_limits:
        df["num_of_outliers"] = outlier_df.sum(axis=1)
        return (
            df,
            number_of_outlier_each_feature_df,
            lower_limit_each_feature_df,
            upper_limit_each_feature_df,
        )

    elif new_feature:
        # add the new feature indicating this row has outlier data to the data and return.
        df["num_of_outliers"] = outlier_df.sum(axis=1)
        return df, number_of_outlier_each_feature_df

    elif return_limits:
        return lower_limit_each_feature_df, upper_limit_each_feature_df

    return number_of_outlier_each_feature_df


def find_features_with_missing_values(data):
    """
    Find which features having missing values.

    :param data: dataset should be a DataFrame object.
    """
    column_names = data.columns
    feature_with_missing_values = column_names[data.isnull().sum() > 0]
    return feature_with_missing_values


def find_high_correlation(data, threshold):
    """Find the correlation between geatures
    :param data: data set
    :param threshold : threshold to pick the features above than this

    :return dict with correlated features

    # how to read the output
    # eg -- {'feature_7': ['feature_6', 0.9402264038780661], ...
    # feature_7 is correlated with feature_6 with 0.9402264038780661 correlation value
    """
    col_corr_dict = {}
    corr_df = data.corr()
    for row in range(len(corr_df.columns)):
        for col in range(
            row
        ):  # the upper half and lower half is same -- so checking to half way is enough
            if corr_df.iloc[row, col] >= threshold:
                # print(corr_df.columns[col], corr_df.index[row])
                col_corr_dict[corr_df.index[row]] = [
                    corr_df.columns[col],
                    corr_df.iloc[row, col],
                ]  # these two values are highly correlated
    return col_corr_dict
