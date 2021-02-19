import pandas as pd

def get_horse_df():
    return pd.read_csv('datasets/final_horses_mergedChip.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_weather_df():
    return pd.read_csv('datasets/final_weather.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_medical_df():
    return pd.read_csv('datasets/final_horseMD_mergedChip.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_consults_df():
    return pd.read_csv('datasets/final_consults.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_colic_df():
    return pd.read_csv('datasets/final_colic.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_labeled_df():
    return pd.read_csv('datasets/final_consults_labeled.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_labeled_weather_colic_df():
    return  pd.read_csv('datasets/labeled_weather_data_koliek.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_labeled_weather_laminitis_df():
    return  pd.read_csv('datasets/labeled_weather_data_hoefbevangen.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_labeled_weather_respiratory_df():
    return  pd.read_csv('datasets/labeled_weather_data_luchtweg.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)

def get_labeled_weather_skin_df():
    return  pd.read_csv('datasets/labeled_weather_data_huid.csv', encoding='ISO-8859-1', na_filter=False, index_col=0)