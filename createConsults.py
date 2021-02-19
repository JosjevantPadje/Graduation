import getDatasets
import pandas as pd

# load the files


# creates a new dataframe with a consultID (horseID+date)
def create_consultID(df, new_index):
    l = zip(df['DierID'], df["DD-MM-YYYY"])
    consult_l = []
    for a, b in l:
        consult_l.append(a+b)
    consult_df = df.copy()
    consult_df[new_index] = consult_l
    return consult_df

# merges rows of df to dict, index of dict is new_index, other items of df are dicts in dict, lists for list_items and if different values occeur in the non-list items.
def df_merge(md_df, new_index, list_items):
    d = dict()
    df = create_consultID(md_df, new_index)
    for index, row in df.iterrows():
        new_ind = row[new_index]
        if new_ind in d:
            for i in row.index:
                if i in list_items:
                    d[new_ind][i].append(row[i])
                elif row[i] != d[new_ind][i]:
                    if isinstance(d[new_ind][i], list) and d[new_ind][i] not in d[new_ind][i]:
                        d[new_ind][i].append(row[i])
                    else:
                        d[new_ind][i] = [d[new_ind][i], row[i]]
        else:
            d[new_ind] = dict()
            for i in row.index:
                if i in list_items:
                    d[new_ind][i] = [row[i]]
                else:
                    d[new_ind][i] = row[i]
    consult_df_new = pd.DataFrame(d).T
    return consult_df_new



def find_temp(texts):
    result = []
    for s in texts:
        newstr = ''.join((ch if ch in '0123456789.,' else ' ') for ch in s)
        listOfNumbers = [i for i in newstr.split()]
        for i in listOfNumbers:
            while i.startswith('.') or i.startswith(','):
                i = i[1:]
            while i.endswith(',') or i.endswith('.'):
                i = i[:-1]
            if i.startswith('3') or i.startswith('4'):
                i = i.replace(',', '.')
                if i.count(".") < 2:
                    i = float(i)
                    if i > 36 and i < 43 and i not in ['42.4', '42.35', '42.33', '42.0', '41.8', '40.05', '39.95', '39.94', '39.12', '38.12','40.0']:
                        result.append(i)
    return result

def add_temp(df):
    result = []
    for i in df['Number^Tekst']:
        result.append(find_temp(i))
    return result

def main():
    horsesMD_df = getDatasets.get_medical_df()
    consults_df = df_merge(horsesMD_df, 'ConsultID', ['Number^Tekst', 'Soort', 'BTW', 'Amount'])
    consults_df['Temp'] = add_temp(consults_df)
    return consults_df
