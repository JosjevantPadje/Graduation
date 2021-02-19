# print('import')
# import createWeatherData
# print('import')
# import mergeChipNumbers
# print('import')
# import createConsults
# import textProcessing
import makeLabeledWeather

# print('start mergeChip')
# horses_df_clean, df_medical_clean = mergeChipNumbers.main()
# print('write mergeChip')
# df_medical_clean.to_csv('datasets/final_horseMD_mergedChip.csv')
# horses_df_clean.to_csv('datasets/final_horses_mergedChip.csv')
# print('start and write weather')
# createWeatherData.main().to_csv('datasets/final_weather.csv')
# print('start and write consults')
# createConsults.main().to_csv('datasets/final_consults.csv')
# print('start textProcessing')
# textProcessing.main().to_csv('datasets/final_consults_labeled.csv')
koliek, hoefbevangen, luchtweg, huid = makeLabeledWeather.main()
koliek.to_csv('datasets/labeled_weather_data_koliek.csv')
hoefbevangen.to_csv('datasets/labeled_weather_data_hoefbevangen.csv')
luchtweg.to_csv('datasets/labeled_weather_data_luchtweg.csv')
huid.to_csv('datasets/labeled_weather_data_huid.csv')
print('Done')
