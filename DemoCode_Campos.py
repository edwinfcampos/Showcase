#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo code on Ensamble Forecasting

This code reads a CSV input file (not included) with hourly forecasting validation data,
then performs a Ridge Normalized Multivariate Regression to output the regression
coefficients into a CSV file for all locations of interest and forecasting lead times

Usage:
./DemoCode_Campos.py -h
./DemoCode_Campos.py '/home/ecampos/repo/output/' '/home/ecampos/repo/output/'
python DemoCode_Campos.py '/home/ecampos/repo/output/' '/home/ecampos/repo/output/'

Input Variables or Arguments:
    input_dir -- Path to inputs directory,
                 For example '/home/ecampos/repo/input/'
    output_dir -- Path to the directory where output will be sent.
                    For example, '/home/ecampos/repo/output/'
    Make sure that these folders already exists in the working directory

Future Improvements:
    This gives an error: "ValueError: Found array with 0 feature(s) (shape=(120, 0)) while a minimum of 1 is required."

Created by Edwin Campos
Last modification on 2018 Dec 6 by ecampos.phd@gmail.com
This code usage is for demo only, and its commercial application is not authorized.
To access this Docstring in the Python console, type 'DemoCode_Campos.__doc__'
"""

# Dependencies
# For most of my codes
import argparse
import datetime
import logging
import sys
# For computation here
# import sframe   # Install with $ pip install -U sframe
import json
import csv
import numpy as np
import os
import pandas as pd
# For machine learnign
from sklearn.linear_model import Ridge   # Install with $ conda install scikit-learn
from sklearn.utils import shuffle

# Global Constant Values
CODE_NAME ='DemoCode_Campos'
WEATHER_VARIABLES = ['precipitation', 'relative_humidity_surf', 'temperature_surf', 'wind_speed_surf']  #['hourlyTemp', 'sustainedWindSpeed']
LEAD_TIMES = [i for i in range(1, 121)]
PROVIDERS = ['schneider', 'accuweather']  # 'nws', 'climatology' will be added in the future # These will be the blending model features
COMMON_COLUMNS = [    # Make sure that forecast_leadtime_column = COMMON_COLUMNS[2] always
    #'_id', # Identification number in mongoDB
    ##'forecast_source',
    'forecast_value',  # Can be 'accuweather_forecast_value', 'nws_forecast_value', or 'schneider_forecast_value'
    'location_id',        # Identification number in Weather Engine, e.g., '0a01c690-ca15-49a0-9689-7d50529ef3ac...'
    #'observation_source',
    'observation_value',  # Can be
    #'time_valid_iso',    # e.g., '2017-06-11 00:00:00'    
    'time_lead',         # in milliseconds, e.g., 3600000. This column will be converted into hours later
    'time_valid',  #in milliseconds, e.g., 1503705600000  
    'variable_name'
    ]
PREDICTOR_STRING = 'forecast_value'
OBSERVATION_STRING = 'observation_value'
RELEVANT_COLUMNS = ['location_id','time_valid',OBSERVATION_STRING] #, PREDICTOR_STRING]

TEST_FRACTION=0.2  # Fraction of the input dataframe to be used as test dataset in the optimization of the lambda factor of the Ridge regression
TRAIN_FRACTION=0.6 # Fraction of the input dataframe to be used as training dataset in the optimization of weighting coefficients of the Ridge regression.
WITHOUT_ZEROS= False  # If True, it will filter out zero values from the dataset (e.g., a row equal to 0.0 mm/h in the precipitation column)
FIT_INTERCEPT = True  # If FALSE, no intercept is used in Ridge regression calculations, and data is expected to be already centered.

# # Function to Trap unvalid Python versions
# def TrapUnvalidPythons():
#     logging.info("Python version: " + sys.version)
#     major_release = int(sys.version.split('.', 1)[0])
#     if (major_release > 2.99):
#         errorString = "This program requires python 2!"
#         logging.error(errorString)
#         sys.exit(errorString)

# Function to set up a file/log to register program behaivoirs
def BuildLogger():
    """Builds a log to register software behaivoir
    Keywords are not required
    """
    logging.basicConfig(
            filename=CODE_NAME+'.log', 
            filemode='w',    # Create the Log file, without including messages from earlier runs
            level=logging.INFO,     # logging.WARNING will record messages labelled as logging.warning or higher.
            format='%(asctime)s. %(levelname)s: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S %Z')   # Displaying the date/time in log messages
    logging.debug('Used to list detailed information, of interest when diagnosing software behavior.')
    logging.info('Used to confirm that software is working as expected.')
    logging.warning('Used to indicate that something unexpected happened, but software is still working as expected.')
    logging.error('Used to indicate that software has not been able to perform some function.')
    logging.critical('Used to indicate that the software itself may be unable to continue running.')
    logging.info('Python version ' + sys.version)


# Methods for generating the Go/No-Go report
class WeatherModel(object):
    def __init__(self, coef_file_path):
        with open(coef_file_path, 'r') as jsonfile:
            self.coefficients = json.load(jsonfile)

    def _get_time_lead_index(self, df):
        time_lead = df['time_lead'].unique()
        if len(time_lead) > 1:
            raise ValueError("Too many lead times at once in call to WeatherModel predict")
        time_lead_ms = time_lead[0] - 3600000
        return self.coefficients['default']['blendingLeadTime'].index(time_lead_ms)

    def _get_variable_name(self, df):
        variable_names = df['variable_name'].unique()
        if len(variable_names) > 1:
            raise ValueError("Too many variables at once in call to WeatherModel predict")
        return variable_names[0]

    def predict(self, df):
        time_lead_index = self._get_time_lead_index(df)
        variable_name = self._get_variable_name(df)

        bias = self.coefficients['default']['variableWeights'][variable_name]['bias']
        schneider_coefficient = \
        self.coefficients['default']['variableWeights'][variable_name]['providerBlending']['schneider'][time_lead_index]
        accuweather_coefficient = \
        self.coefficients['default']['variableWeights'][variable_name]['providerBlending']['accuweather'][
            time_lead_index]

        numerator = schneider_coefficient * df['schneider_forecast_value'] + accuweather_coefficient * df[
            'accuweather_forecast_value']
        denominator = schneider_coefficient + accuweather_coefficient

        return (numerator / denominator)

    def predict_manual(self, df, schneider_coefficient, accuweather_coefficient):

        if abs(schneider_coefficient + accuweather_coefficient - 1) > 0.001:
            raise ValueError(
                "Schneider and Accuweather Coefficients must be normalized before passing to predict_manual")

        numerator = schneider_coefficient * df['schneider_forecast_value'] + accuweather_coefficient * df[
            'accuweather_forecast_value']
        denominator = schneider_coefficient + accuweather_coefficient

        return (numerator / denominator)


def many_ridge_regressions(data_train, data_test, predictors, reference, alpha):
    """Method to fit multivariate regression models using different Ridge regularization coefficients

    Arguments:
    data_train: Dataframe with values to be used to independently train the Ridge Regularization model
    data_test: Dataframe with values to be used to independently test the Residual Sum of Squares metric
    predictors: array of model features, as alphanumeric values corresponding to the columns of data
    reference: alphanumeric value corresponding to the reference column of the dataframe (e.g., observations)
    alpha: A coefficient of Ridge regularization;
           e.g., 1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, or 20

    Outputs:
        ret: Array with [alpha, residuals-square-sum, regression interpcept, regression coefficients]

    Dependencies:
    import numpy, Pandas, and sklearn

    Modified from
    https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/

    """
    ridgereg = Ridge(alpha=alpha, fit_intercept=FIT_INTERCEPT)  # IF no intercept is used in calculations (e.g. fit_intercept=False, and data is expected to be already centered).

    # Convert Pandas data frame inputs to numpy.ndarray, which is what sklearn uses
    data_reference_numpy = data_train[reference].values.astype('float')  # Make sure this gives no strings
    data_predictors_numpy = data_train[predictors].values.astype('float')  # Make sure this gives no strings
    data_test_reference_numpy = data_test[reference].values.astype('float')
    data_test_predictors_numpy = data_test[predictors].values.astype('float')

    if np.isinf(data_predictors_numpy).any():
        print('ERROR: data_predictors_numpy has INFINITES values')
        print(data_predictors_numpy)
    elif np.isinf(data_reference_numpy).any():
        print('ERROR: data_reference_numpy has INFINITES values')
        print(data_reference_numpy)
    elif np.isnan(data_predictors_numpy).any():
        print('ERROR: data_predictors_numpy has NOT-A-NUMBER values')
        print(data_predictors_numpy)
    elif np.isnan(data_reference_numpy).any():
        print('ERROR: data_reference_numpy has NOT-A-NUMBER values')
        print(data_reference_numpy)
    elif 0 in data_predictors_numpy.shape:
        print('ERROR: data_predictors_numpy has a feature with zero values')
        print( "data_predictors_numpy.ndim = %i" % data_predictors_numpy.ndim )
        print("data_predictors_numpy.shape = ", data_predictors_numpy.shape)
        print('data:', data)

    # Obtain the Ridge Regularization regression model
    ridgereg.fit(data_predictors_numpy, data_reference_numpy)

    # Evaluate the Residual Sum of Squares for the above model using an independent dataset
    y_pred = ridgereg.predict(data_test_predictors_numpy)

    # Return the result in pre-defined format
    ret = [alpha]
    rss = sum((y_pred - data_test_reference_numpy) ** 2)
    ret.extend([rss])
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret


# Subsample a Pandas dataframe into Test, Training, and Validation datasets (NEEDS TO BE TESTED)
def SplitPandasIntoTestTrainingValidation(input_pandas_dataframe, test_fraction=0.2, train_fraction=0.6):
    """Randomly splits a Pandas dataframe into training (default 60% of full dataset),
    test (20% of full dataset), and validation dataframes (remaining of full dataset).

    Keyword arguments:
    input_pandas_dataframe -- Original Pandas dataframe with input values
    test_fraction -- default=0.2,
    training_fraction -- default=0.6
            #train_data,test_data = oneleadtime_relevant_dataframe.random_split(.8,seed=0)  #
            This works with SFrames but not with Pandas

    Outputs:
    test_dataframe -- Pandas dataframe with the test_fraction of the input_pandas_dataframe
    train_dataframe -- Pandas dataframe with the train_fraction of the input_pandas_dataframe
    validate_dataframe -- Pandas dataframe with the remaining fraction of the input_pandas_dataframe
    """
    # TEST: Make sure that the test and train fraction are smaller or equal to the 100% of the sample
    if (test_fraction + train_fraction) > 1:
        raise Exception(
            'TEST1 ERROR: Invalid inputs for test_fraction & train_fraction in SplitPandasIntoTestTrainingValidation')
    # TEST: Make sure that the input dataframe is not empty
    if input_pandas_dataframe.shape[0] < 1:
        raise Exception('TEST2 ERROR: input_pandas_dataframe is empty in SplitPandasIntoTestTrainingValidation')

    # print(input_pandas_dataframe.shape[0])
    shuffled_df = shuffle(input_pandas_dataframe)
    row_count = shuffled_df.shape[0]
    split_point_test = int(row_count * test_fraction)
    #    print('split_point_test:', split_point_test)
    test_dataframe = shuffled_df[:split_point_test]
    #    print('# rows of test_dataframe:',test_dataframe.shape[0])
    split_point_training = int(row_count * (1.0 - train_fraction))
    #    print('split_point_training:',split_point_training)
    train_dataframe = shuffled_df[split_point_training:]
    #    print('# rows of train_dataframe:',train_dataframe.shape[0])
    validate_dataframe = shuffled_df[split_point_test:split_point_training]
    #    print('# rows of validate_dataframe:',validate_dataframe.shape[0])

    return test_dataframe, train_dataframe, validate_dataframe


# Filter out invalid values
def EliminateInvalidRowsOrColumns(a_weather_variable, input_pandasframe, without_zeros=True):
    """Method to check if we have rows or columns to get rid of

    Arguments:
        a_weather_variable -- Identifies the column to use
        input_pandasframe -- Pandas data frame with merged values
        without_zeros -- Optional, to filter out zero values from the sample
                     (e.g., rows in precipitation columns with values = 0 mm/h)
    Also needs a definition of the global variable RELEVANT_COLUMNS

    """
    #print(input_pandasframe.columns)
    #relevant_cols = [col for col in input_pandasframe.columns if RELEVANT_COLUMNS in col]
    relevant_cols = [col for col in input_pandasframe.columns if (col in RELEVANT_COLUMNS) or (PREDICTOR_STRING in col)]
    ##relevant_cols = [col for col in input_pandasframe.columns if col not in IRRELEVANT_COLS]
    # logging.info('input_pandasframe columns in EliminateInvalidRowsOrColumns=%s' % input_pandasframe.columns)
    # logging.info('relevant_cols in EliminateInvalidRowsOrColumns=%s ' % relevant_cols)

    for column in relevant_cols:
    #for column in RELEVANT_COLUMNS:
        relevant_dataframe = input_pandasframe
        logging.debug(column)
        logging.debug(relevant_dataframe[column][0:20])

        # Replace strings (such as 'None') with None values
        relevant_dataframe.replace('None', np.nan, inplace=True)  # It's OK to get a SettingWithCopyWarning here.

        # Exclude rows that have cells with '' or None values
        # relevant_dataframe = relevant_dataframe.filter_by(invalid_entries, column, exclude=True)  # This gives TypeError: Type of given values does not match type of column 'observation_value_campos' in SFrame.
        relevant_dataframe = relevant_dataframe.dropna(subset=[column], how='any')  # Remove the None values

        if len(relevant_dataframe) == 0:  # Delete entire column if there are no useful values
            logging.warning(
                'The column %s in relevant_dataframe will be deleted because all its values are None or NaN.' % column)
            ##input_pandasframe.remove_column(column)  # This will not modify the original dataframe
            input_pandasframe = input_pandasframe.drop(labels=column, axis=1)
            continue  # Continues with the next iteration of the "for column..." loop
        elif len(relevant_dataframe) == len(input_pandasframe):  # Delete only rows with non-useful values
            logging.info('Column %s, is free from NaN or None values' % column)
        else:
            logging.info('In column %s, we filtered out %i unvalid rows (with None or NaN values)' % (
                column, len(input_pandasframe) - len(relevant_dataframe)))

            # Check if we have zero values to get rid of
        if without_zeros:
            logging.info(
                'The EliminateInvalidRowsOrColumns function is checking if there are zero values to get rid of')
            relevant_dataframe_filtered = relevant_dataframe[relevant_dataframe[column] != 0.0]
            if len(relevant_dataframe_filtered) != len(relevant_dataframe):
                relevant_dataframe = relevant_dataframe_filtered
                logging.warning('In column %s, there were %i unvalid rows (with values equal to ZERO)' % (
                    column, len(relevant_dataframe) - len(relevant_dataframe_filtered)))
                logging.warning('These were cleaned, and the number of rows changed from %i to %i.' % (
                    len(relevant_dataframe), len(relevant_dataframe_filtered)))
            else:
                logging.info('Column %s, is free from values equal to ZERO.' % column)
        else:
            logging.info('The EliminateInvalidRowsOrColumns function will keep all zero values.')

        input_pandasframe = relevant_dataframe

        # logging.info(relevant_dataframe[column][0:20])
    # logging.info('Dataframe cleaned by EliminateInvalidRowsOrColumns:{}'.format(input_pandasframe))

    return input_pandasframe

# Methods to match observations and predictions
class PivotTree(object):
    def __init__(self):
        self.tree = {}

    def insert_one(self, matched_dict):
        """Insert one matched_dict into the Pivot Tree

        Args:
            matched_dict (Dict): a matched dict e.g.
                {'observation_source': 'campos',
                 'location_id': 'e933542e-dc83-4fe4-953c-c7ad7cc7ed4d',
                 'forecast_value': 10.0,
                 'forecast_source': 'batman',
                 'time_lead': 7200000,
                 'variable_name': 'precipitation',
                 'time_valid': 1523750400000,
                 'observation_value': 0
                }
        Return:
            None
        """

        lid = matched_dict.get('location_id')
        time_valid = str(matched_dict.get('time_valid'))
        forecast_source = matched_dict.get('forecast_source')

        if self.tree.get(lid) is None:
            self.tree[lid] = {}

        if self.tree[lid].get(time_valid) is None:
            self.tree[lid][time_valid] = {}

        if self.tree[lid][time_valid].get(forecast_source) is not None:
            print("Duplicate Forecast/Observation pair found at ({},{},{})".format(lid, time_valid, forecast_source))

        self.tree[lid][time_valid][forecast_source] = matched_dict

    def insert_many(self, match_list):
        """Insert a list of matched dicts to PivotTree

        Args:
            match_list (list of Dicts): a list to insert

        Return:
            None
        """
        for p in match_list:
            self.insert_one(p)

    def values(self):
        for lid, i in self.tree.items():
            for tv, j in i.items():
                yield self._make_row(lid, tv, j)

    def _get_common_leaf_field(self, leaf_dict, fieldname):
        l = [i.get(fieldname) for i in leaf_dict.values() if i.get(fieldname) is not None]
        return l[0] if len(l) > 0 else None

    def _make_row(self, location_id, time_valid, leaf_dict):

        return {
            'location_id': location_id,
            'time_lead': self._get_common_leaf_field(leaf_dict, 'time_lead'),
            'variable_name': self._get_common_leaf_field(leaf_dict, 'variable_name'),
            'time_valid': int(time_valid),
            'observation_value': self._get_common_leaf_field(leaf_dict, 'observation_value'),
            'schneider_forecast_value': leaf_dict.get('schneider', {}).get('forecast_value'),
            'nws_forecast_value': leaf_dict.get('nws', {}).get('forecast_value'),
            'accuweather_forecast_value': leaf_dict.get('accuweather', {}).get('forecast_value')
         }

    def toDF(self):
        return pd.DataFrame(self.values())


# Main function to call submethods for computing coefficients for Blending algorithm
def ComputeBlendCoefficientsCommand(input_dir, output_dir):
    """Call functions that compute blending forecasting coefficients.

    Arguments:
    input_dir -- Path to inputs directory,
                 For example '../input/'
    output_dir -- Path to the directory where output will be sent.
                    For example, '../output/'

    """
        
    # Prepare outputs
    outputFile = output_dir+CODE_NAME+'.csv'
    message = 'Preparing the output file:%s' % outputFile
    logging.info(message)
    
    output_target = open(outputFile, 'w')
    writer = csv.writer(output_target,delimiter=",")
    
    #Initialize predictors
    predictors= ['coef_%s'%prov for prov in PROVIDERS]
    headerlist_to_print = ['one_weather_variable','time_lead','intercept'] + predictors
    logging.info(headerlist_to_print)
    writer.writerow(headerlist_to_print)

    factor_2convert_milliseconds_into_hours = (1.0 / 1000.0) * (1.0 / 60.0) * (1.0 / 60.0)

    for one_weather_variable in WEATHER_VARIABLES:
        logging.info('one_weather_variable=%s' % one_weather_variable)

        # Create Pandas dataframe with output dataset: the heterogeneous blending coefficients
        ind = range(0, len(LEAD_TIMES))
        train_coef_matrix_ridge = pd.DataFrame(index=ind, columns=headerlist_to_print)

        index_lead_time = 0
        for one_lead_time in LEAD_TIMES:

            input_filename = "es_matched_forecasts-{}-{}.json".format(one_weather_variable, one_lead_time)
            with open(os.path.join(input_dir, input_filename), "r") as jsonfile:
                a = json.load(jsonfile)
                if len(a) == 0:
                    logging.info("Empty File: {}, skipping...".format(input_filename))
                    continue

            pt = PivotTree()
            pt.insert_many(a)

            merged_dataframe = pt.toDF()  # Pandas Dataframe
            logging.info(merged_dataframe)

            logging.info(merged_dataframe.columns)
            if len(merged_dataframe) == 0:  # Empty Pandas dataframe
                logging.info('The merged_dataframe was empty for one_lead_time=$i' % one_lead_time)
                continue  # Go to the next iteration of the enclosing for one_lead_time loop

            oneleadtime_relevant_dataframe = EliminateInvalidRowsOrColumns(
                one_weather_variable, merged_dataframe, without_zeros=WITHOUT_ZEROS)
            logging.info('There are %i rows in oneleadtime_relevant_dataframe.' % len(oneleadtime_relevant_dataframe))

            lead_time_string = 'lead_time = %i milliseconds or %0.1f hours' % (
            one_lead_time, one_lead_time * factor_2convert_milliseconds_into_hours)
            logging.info('There are %i rows in oneleadtime_relevant_dataframe when using %s.' % (
            len(oneleadtime_relevant_dataframe), lead_time_string))

            if len(oneleadtime_relevant_dataframe) <= 1: # Empty data frame
                logging.info(
                        'The oneleadtime_relevant_dataframe was empty for one_lead_time = %s hrs' % 
                        str(one_lead_time*factor_2convert_milliseconds_into_hours) )
                continue # Since there are no useful data for this lead time, continue with the next iteration of the loop

            # Subsample data into Test, Training, and Validation datasets
            # Split the dataset into training (80% of full dataset) and test (20% of full dataset)
            # train_data,test_data = oneleadtime_relevant_dataframe.random_split(.8,seed=0)  # This works with SFrames but not with Pandas
            # TEST_FRACTION=0.2 is defined above
            # TRAIN_FRACTION=0.6 is defined above
            test_data, training_data, validation_data = SplitPandasIntoTestTrainingValidation(
                oneleadtime_relevant_dataframe,
                test_fraction=TEST_FRACTION,
                train_fraction=TRAIN_FRACTION
                )

            logging.info('TEST: #rows(test_data)/#rows(oneleadtime_relevant_dataframe):'.format(
                              float(len(test_data))/len(oneleadtime_relevant_dataframe)) )
            logging.info('TEST: #rows(training_data)/#rows(oneleadtime_relevant_dataframe):'.format(
                              float(len(training_data))/len(oneleadtime_relevant_dataframe)) )
            logging.info('TEST:#rows(validation_data)/#rows(oneleadtime_relevant_dataframe):'.format(
                              float(len(validation_data))/len(oneleadtime_relevant_dataframe)) )

            # Unit test if this worked
            # TEST: If the sample size is correct for each dataframe
            logging.info('TEST: Is TEST_FRACTION = #rows(test_data)/#rows(oneleadtime_relevant_dataframe)?' +
                         str(abs(
                             TEST_FRACTION - (float(len(test_data)) / len(oneleadtime_relevant_dataframe))) < 0.01)
                         )
            logging.info('TEST: Is TRAIN_FRACTION = #rows(training_data)/#rows(oneleadtime_relevant_dataframe)?' +
                         str(abs(TRAIN_FRACTION - (
                         float(len(training_data)) / len(oneleadtime_relevant_dataframe))) < 0.01)
                         )
            logging.info(
                'TEST: Is (1-TEST_FRACTION-TRAIN_FRACTION) = #rows(validation_data)/#rows(oneleadtime_relevant_dataframe)? ' +
                str(abs((1.0 - TEST_FRACTION - TRAIN_FRACTION) - (
                float(len(validation_data)) / len(oneleadtime_relevant_dataframe))) < 0.01)
                )

            # Use Test dataset to obtain optimal penalty
            # Set the different values of alpha to be tested
            alpha_ridge = [0.0, 1e-20, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
            # Find list of strings with names of predictors columns with valid values: data frame column names to be used as predictors
            predictors_valid = [predictor_column_name for predictor_column_name in
                                oneleadtime_relevant_dataframe.columns if
                                PREDICTOR_STRING in predictor_column_name and       # Is this column corresponding to one of the predictions
                                any(x in predictor_column_name for x in PROVIDERS)  # Is this column corresponding to one of the providers
                                ]
            logging.info('oneleadtime_relevant_dataframe.columns = %s' % oneleadtime_relevant_dataframe.columns)
            logging.info('predictors_valid = %s' % predictors_valid)   # ['accuweather_forecast_value', 'nws_forecast_value', 'schneider_forecast_value']
            # Find string with name of observations column
            observations = [obs_column_name for obs_column_name in
                            oneleadtime_relevant_dataframe.columns if
                            OBSERVATION_STRING in obs_column_name
                            #obs_column_name in OBSERVATION_STRING
                            ]
            if len(observations) == 0:
                logging.warning('There are no valid observations for %s at leadtime= %s' %
                                (one_weather_variable, one_lead_time)
                                )
                continue  # Go to the next iteration of the enclosing for one_lead_time loop

            observations = observations[0] # str(observations).strip("'[]'")
            logging.info('observations column name = %s' % observations)            
            # Initialize the dataframe for storing coefficients.
            # Recall that PREDICTORS = ['nws', 'schneider', 'accuweather' ]
            col = ['alpha','rss','intercept']
            for one_predictors_valid in predictors_valid:
                col += ['coef_%s'%prov for prov in PROVIDERS if prov in one_predictors_valid ]
            logging.info('col= %s' % col)      
            
            ind = range(0,len(alpha_ridge))
            coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)   # Recall that ind: # of lead times, col: # of predictors
            for index_alpha_ridge in range(len(alpha_ridge)):
                coef_matrix_ridge.iloc[index_alpha_ridge,] = many_ridge_regressions(
                                                  training_data,
                                                  test_data,
                                                  predictors_valid, 
                                                  observations, 
                                                  alpha_ridge[index_alpha_ridge])            
            print(one_weather_variable, coef_matrix_ridge)
            rss_min = coef_matrix_ridge['rss'].min()
            optimal_alpha = coef_matrix_ridge['alpha'][coef_matrix_ridge['rss']==rss_min].iloc[0]
            logging.info('The minumum Residuals-Square-Sum of %s corresponds to alpha = %s' % (rss_min, optimal_alpha) )

            # Train the blending model with forecast providers (features)            
            # Fit the model
            ridge_regression = Ridge(alpha=optimal_alpha,fit_intercept=FIT_INTERCEPT)  # If no intercept will be used in calculations, then data is expected to be already centered.     
            # Convert Pandas data frame inputs to numpy.ndarray, which is what sklearn uses
            data_reference_numpy = training_data[observations].values.astype('float')
            data_predictors_numpy = training_data[predictors_valid].values.astype('float')

            ridge_regression.fit(data_predictors_numpy,data_reference_numpy)
            expected_value_from_regression = ridge_regression.predict(data_predictors_numpy)

            if FIT_INTERCEPT is False:
                # COMPUTE MEAN ERRORS WITH VALIDATION DATASET, AND USE THESE AS ridge_regression.intercept_
                linear_bias_coefficient_numpyarray = np.subtract(              # Subtract arguments, element-wise
                                                                expected_value_from_regression, 
                                                                data_reference_numpy
                                                                )  
                linear_bias_coefficient = linear_bias_coefficient_numpyarray.sum()
                #linear_bias_coefficient = ridge_regression.intercept_
            else:
                linear_bias_coefficient = ridge_regression.intercept_

            # Output CSV file with validation data for Go/NoGo report
            validation_data_predictors_numpy = validation_data[predictors_valid].values.astype('float')
            validation_data['challenger_prediction'] = ridge_regression.predict(validation_data_predictors_numpy)
            weather_model = WeatherModel('/home/ecampos/repo/input/weights_prod20171205.js')
            validation_data['incumbent_prediction'] = weather_model.predict(validation_data)

            validation_data.to_csv(
                output_dir + 'validation-' + one_weather_variable + '-' + str(one_lead_time) + '.csv')

            # Build the lead-time row in the proper format
            # Recall that headerlist_to_print = ['one_weather_variable', 'time_lead', 'intercept', 'coef_nws', 'coef_schneider', 'coef_accuweather']
            # Recall that predictors_valid = ['forecast_value_schneider', 'forecast_value_accuweather'] and some times also 'forecast_value_nws'
            buff_coef_matrix_ridge = [one_weather_variable, one_lead_time, linear_bias_coefficient]
            for column_index in range( 3, len(headerlist_to_print) ):
                which_provider = headerlist_to_print[column_index][5:]
                logging.info( "column_index:{}, which_provider:{}".format(column_index, which_provider) )
                new_entry = 0.0   # Default value
                # Check if the trainned model used forecasts from this provider 
                for index,one_predic in enumerate(predictors_valid):
                    #print(headerlist_to_print[column_index], which_provider, one_predic)
                    if which_provider in one_predic:
                        new_entry = ridge_regression.coef_[index]
                        #print('This is a match and the coefficient will be %f' % new_entry)
                #buff_coef_matrix_ridge.extend(new_entry)  # This does NOT work within a loop
                #buff_coef_matrix_ridge += new_entry  # This does NOT work within a loop
                #print('The value of new_entry is %f' % new_entry)
                buff_coef_matrix_ridge.append(new_entry)
            print(buff_coef_matrix_ridge)
            writer.writerow(buff_coef_matrix_ridge)  # Write dataframe row info into a CSV file
            logging.info(index_lead_time)
            train_coef_matrix_ridge.iloc[index_lead_time,] = buff_coef_matrix_ridge
            index_lead_time += 1 
        #train_coef_matrix_ridge.append(buff_coef_matrix_ridge)
        # If the above gives "UnboundLocalError: local variable 'buff_coef_matrix_ridge' referenced before assignment"
        # Then the input data is not useful for deriving blending coefficients. Check PrepareBlendFcstInputsFromElasticSearchDB.py code
        logging.info(train_coef_matrix_ridge)
    logging.warning('Closing %s' % outputFile)
    output_target.close()


# Pass DemoCode_Campos.py script arguments to PrepareBlendFcstCommand function, only if this code is run standalone
if __name__ == '__main__':   # For example: '$ python DemoCode_Campos.py'
    #import datetime
    startTime = datetime.datetime.now()

    BuildLogger()
    logging.info('DemoCode_Campos.py is being run.')
    #print('DemoCode_Campos.py is being run.')
    #TrapUnvalidPythons()

    #import argparse
    parser = argparse.ArgumentParser(
        description='Outputs a dataset for Blending Forecast Algorithm')  # By default prog=sys.argv[0] = 'DemoCode_Campos.py'
    parser.add_argument('input_dir',
                        help = "Path to input directory, for example '../input/' ")
    parser.add_argument('output_dir',
                        help="Path to output directory, for example '../output/' ")
    args = parser.parse_args()

    ComputeBlendCoefficientsCommand(**vars(args)) # Note that type(args) is a Namespace, and type(vars(args)) is a dictionary

    endTime = datetime.datetime.now()
    logging.info('This code ran in ' + str( (endTime - startTime).seconds/60.0 ) + ' minutes.')
    print('This code ran in ', (endTime - startTime).seconds/60.0, ' minutes.')


else:     # For example '$ python' and  '>>> import DemoCode_Campos'
    BuildLogger()
    logging.info('DemoCode_Campos.py has being imported from another module.')
    #print('DemoCode_Campos.py has being imported from another module.')
    #TrapUnvalidPythons()
