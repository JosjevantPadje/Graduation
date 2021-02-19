from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


#To test the KNNimputer and to test how many neighors we can use best.
def test_KNNimputer(df, df_target, min, max):
    df.Date = pd.to_datetime(df.Date, format='%d-%m-%Y')
    df = df.set_index('Date')
    r2s = []
    for i in range(min, max+1):
        imputer = KNNImputer(n_neighbors=i)
        df_filled = imputer.fit_transform(df).T
        df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns.values)
        df_filled = df_filled.set_index(df.index)

        r2 = r2_score(df_target['DR'], df_filled['DR'])
        r2s.append((r2, i))
    s = sorted(r2s)
    return s[len(s)-1][1]



# Load csv - "datasets/heino.csv"
def load_csv(csv):
    df = pd.read_csv(csv, skipinitialspace=True)
    # Set date as index
    df['Date'] = pd.to_datetime(df.YYYYMMDD, format='%Y%m%d')
    df = df.set_index('Date')
    return df


# Remove useless/empty rows
def remove_rows(r, df):
    for i in r:
        del df[i]
    return df

# Impute the missing values
def impute(neighbors, df): #neighbors = 10
    imputer = KNNImputer(n_neighbors=neighbors)
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns.values)
    df_filled = df_filled.set_index(df.index)
    return df_filled


# Add missing rows from Hoogeveen into Heino
def add_values(v, df1, df2):
    for i in v:
        df1[i] = df2[i]
    return df1

def average(lst):
    return sum(lst) / float(len(lst))

def getAverage(df, new_df, x, i, c):
    avglist = []
    for j in range(0, x):
        if i - j in df.index:
            avglist.append(df.loc[i - j, c])
    new_df.loc[i, c] = average(avglist)
    return new_df

def getShift(df, new_df, x, i, c):
    if i - x in df.index:
        new_df.loc[i, c] = df.loc[i-x, c]
    return new_df

def renameColumns(new_df, x):
    columns = []
    for c in new_df.columns:
        columns.append(c+str(x))
    new_df.columns = columns
    new_df.drop('Date'+str(x), axis=1, inplace=True)
    return new_df

def addNewRows(df):
    df.reset_index(level=0, inplace=True)
    df30 = df.copy()
    df14 = df.copy()
    df4 = df.copy()
    df3 = df.copy()
    df2 = df.copy()
    df1 = df.copy()
    for i, r in df.iterrows():
        for c in df.columns:
            if c != 'Date':
                df30 = getAverage(df, df30, 30, i, c)
                df14 = getAverage(df, df14, 14, i, c)
                df4 = getShift(df, df4, 4, i, c)
                df3 = getShift(df, df3, 3, i, c)
                df2 = getShift(df, df2, 2, i, c)
                df1 = getShift(df, df1, 1, i, c)
    df30 = renameColumns(df30, 30)
    df14 = renameColumns(df14, 14)
    df4 = renameColumns(df4, 4)
    df3 = renameColumns(df3, 3)
    df2 = renameColumns(df2, 2)
    df1 = renameColumns(df1, 1)
    df_complete = pd.concat([df, df30, df14, df4, df3, df2, df1], axis = 1, sort=False)
    df_complete = df_complete.set_index('Date')
    return df_complete

def main():
    vars_to_remove = ['YYYYMMDD', 'STN', 'PG', 'PX', 'PXH', 'PN', 'PNH', 'VVN', 'VVNH', 'VVX', 'VVXH', 'NG', 'FHXH',
                      'FHNH', 'FXXH', 'TNH', 'TXH', 'T10NH', 'RHXH', 'UXH', 'UNH']
    missing_variables = ['PG', 'PX', 'PN']

    df = pd.read_excel("datasets/Heino_compleet.xlsx", sheet_name="value")
    df_target = pd.read_excel("datasets/Heino_compleet.xlsx", sheet_name="target")

    neighbors = test_KNNimputer(df, df_target, 1, 20)
    heino_df = load_csv("datasets/heino.csv")
    heino_df_removed = remove_rows(vars_to_remove, heino_df)
    heino_df_filled = impute(neighbors, heino_df_removed)

    # Read Hoogeveen CSV data make similar format as Heino
    hoogeveen_df = load_csv("datasets/hoogeveen.csv")
    df_new = add_values(missing_variables, heino_df_filled, hoogeveen_df)
    return addNewRows(df_new)
