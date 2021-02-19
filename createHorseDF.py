import pandas as pd

# Load dataframes from csv
animal_df = pd.read_csv("datasets/ExportDieren.csv", encoding='ISO-8859-1', na_filter=False)
md_animal_df = pd.read_csv("datasets/ExportMD.csv", encoding='ISO-8859-1', na_filter=False)

# Get indeces of animals that have are not horse
indexNames = animal_df[ animal_df['Soort'] != 'Paard' ].index
# Remove non horse indices from dataframe
horse_df = animal_df.drop(indexNames, inplace=False)

# Get list of horse ID's
horseIds = list(horse_df['AnimalID'])

# Get indices of medical lines that are not horses
indexNames = md_animal_df[~md_animal_df['DierID'].isin(horseIds)].index
# Remove non horse intries from dataframe
md_horse_df = md_animal_df.drop(indexNames, inplace=False)
