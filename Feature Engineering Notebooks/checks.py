import datetime
import pandas as pd

def test_dtypes_plugin_vals(df):
    # check date-time format
    assert isinstance(df['start'].iloc[0], datetime.datetime), \
    "Make sure your time vars are converted to datetime."
    assert isinstance(df['end'].iloc[0], datetime.datetime), \
    "Make sure your time vars are converted to datetime."
    # check that all values are 1.0 --> indicating a plug-in state
    assert df['value'].all(), \
    "Your DataFrame should hold rows from the plug-in event stream only, when a device is plugged in."
    # check that an additional col (6 instead of 5) has been added
    assert len(df.columns) > 5, \
    "Make sure you've added an additional charge duration column."
    # tests pass in green!
    print('\033[92m'+'\033[1m'+"Passed format and 'duration' column tests so far!")
    

def noise_tests(df):
    # check that there are no >2021 dates
    end_of_2021 = pd.Timestamp(2021, 12, 30, 0)
    assert (df['end'] < end_of_2021).all(), \
    'Your data should only include start/end times that have happened in the past.'
    # check that there or no NaN values
    assert df['start_batt_level'].isnull().sum() == 0, \
    'Your data should NOT contain any null values.'
    # code at start is for green + bold!
    print('\033[92m'+'\033[1m'+'Passed all noise removal tests, great work!')