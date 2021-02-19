import createHorseDF
import numpy as np
from collections import Counter
from difflib import SequenceMatcher

# Create a list of the chip numbers that
# exisit 2 or 3 times in the df
# and have a minimum length of 6
def get_duplicates(df):
    horses_dict = dict(Counter(df['Chip nummer']))
    dups = []
    for key, value in horses_dict.items():
        if value > 1 and value < 4 and len(key) > 5:
            dups.append(key)
    return dups

# Removes items from list
def remove_if(i, l):
    if i in l:
        l.remove(i)
    return l

# Returns a list with the longest strings if
# two strings are more then 75% alike or
# the one string fits completely into the other
def name_color_breed(objects):
    results = objects.copy()
    for i in objects:
        for j in objects:
            if i < j:
                if SequenceMatcher(None, i.lower(), j.lower()).ratio() > 0.75:
                    if i == j:
                        results = results
                    elif len(j) > len(i):
                        results = remove_if(i, results)
                    else:
                        results = remove_if(j, results)
                elif i.lower() in j.lower():
                    results = remove_if(i, results)
                elif j.lower() in i.lower():
                    results = remove_if(j, results)
    return results

# removes the 'O' (unknown) if other genders are known, sorts the genders
def geslacht(objects):
    if len(objects) > 1 and "O" in objects:
        objects.remove('O')
    if "M" in objects and ('MG' in objects or "H" in objects):
        objects = ['O']
    if 'MG' in objects and 'H' in objects:
        objects = ['MG']
    return sorted(objects)

# if one of the entries died, became inactive or is deleted, the merged horse is saved like that
def overleden_inact_del(l):
    if 1 in l:  # horses_df.loc[indices[0]][c]:
        l = [1]
    return l

# if one or more birthdates start with 01-01 while other not 01-01 dates are available, the 01-01 dates are removed
def geboortedatum(l):
    while np.nan in l:
        l.remove(np.nan)
    res = []
    for i in l:
        res.append(i.startswith('01-01-'))
    if True in res and False in res:
        lc = l.copy()
        for b, v in zip(res, lc):
            if b:
                l.remove(v)
    return l

# removes unknown, nan and '' from the list
def clean_list(l):
    while 'unknown' in l:
        l.remove('unknown')
    while np.nan in l:
        l.remove(np.nan)
    while '' in l:
        l.remove('')
    return l

# Creates the complete horses using the functions above and removes the duplicates.
def remove_duplicates(df, df_medical):
    animalIDs = []
    clientIDs = []
    dups = get_duplicates(df)
    df_result = df.copy()
    for chipnr in dups:
        indices = []
        for index, row in df.iterrows():
            if chipnr == row['Chip nummer']:
                indices.append(index)
        for c in df.columns:
            objects = [df.loc[indices[0]][c]]
            for i in range(1, len(indices)):
                objects.append(df.loc[indices[i]][c])
            objects = list(set(objects))
            objects = clean_list(objects)
            if c == 'Naam' or c == 'Kleur' or c == 'Ras':
                objects = name_color_breed(objects)
            elif c == 'Geslacht':
                objects = geslacht(objects)
            elif c == 'Overleden' or c == 'Inact' or c == 'Del':
                objects = overleden_inact_del(objects)
            elif c == 'Geb.Datum':
                objects = geboortedatum(objects)
            if len(objects) == 1:
                result = objects[0]
            elif len(objects) == 0:
                result = ''
            elif c == 'AnimalID':
                result = ', '.join([str(item) for item in sorted(objects)])
                animalIDs.append(result)
            elif c == 'ClientID':
                result = ', '.join([str(item) for item in sorted(objects)])
                clientIDs.append(result)
            else:
                result = ', '.join([str(item) for item in objects])
            df_result.at[indices[0], c] = result
        for i in range(1, len(indices)):
            df_result.drop(indices[i], inplace=True)
    for index, row in df_medical.iterrows():
        for a, c in zip(animalIDs, clientIDs):
            if row['DierID'] in a:
                df_medical.at[index, 'DierID'] = a
                df_medical.at[index, 'ClientID'] = c
    return df_result, df_medical



def main():
    horses_df = createHorseDF.horse_df
    medical_df = createHorseDF.md_horse_df
    horses_df_clean, df_medical_clean = remove_duplicates(horses_df, medical_df)
    return horses_df_clean, df_medical_clean


