import getDatasets
from datetime import datetime

def createData(disease, weather_data, labels):
    weather_parameters = {'koliek': ['FHX', 'FHVEC14', 'FHVEC30', 'FG14', 'FG30', 'FHX14', 'FHX30', 'FHN30', 'TX4', 'SQ2', 'SQ4', 'SQ14', 'SQ30', 'T10N30', 'DR30', 'RH14', 'RH30', 'RHX30',
                                     'UG', 'UG1', 'UG2','UG3','UG4','UG14','UG30','UX30', 'UN', 'UN1','UN2','UN3','UN4','UN14','UN30','PG2', 'PG3', 'PG4', 'PG14', 'PX14', 'PN2', 'PN3', 'PN4', 'PN14'],
                          'hoefbevangen': ['FHVEC','FHVEC1','FHVEC2','FHVEC3','FHVEC4','FHVEC14','FHVEC30', 'FG','FG1','FG2','FG3','FG4','FG14','FG30', 'FHN','FHN1','FHN2','FHN3','FHN4','FHN14',
                                           'FHN30',  'FHX3','FHX4','FHX14','FHX30',
                                           'FXX14', 'FXX30', 'TG','TG1','TG2','TG3','TG4','TG14','TG30','TN','TN1','TN2','TN3','TN4','TN14','TN30','TX','TX1','TX2','TX3','TX4','TX14','TX30',
                                           'T10N', 'T10N1', 'SQ','SQ1','SQ2','SQ3','SQ4','SQ14','SQ30', 'SP','SP2','SP3','SP4','SP14','SP30','Q','Q1','Q2','Q3','Q4','Q14','Q30',
                                           'DR2', 'DR14', 'DR30', 'RH14', 'UG','UG1','UG2','UG3','UG4','UG14','UG30','UN','UN1','UN2','UN3','UN4','UN14','UN30','UX14', 'UX30','PX3',
                                           'EV24','EV241','EV242','EV243','EV244','EV2414','EV2430'],
                          'luchtweg': ['FHVEC', 'FHVEC14', 'FHVEC30', 'FG', 'FG14', 'FG30', 'FHX', 'FHX30', 'FHN','FHN1','FHN2','FHN4','FHN14','FHN30', 'TG','TG1','TG2','TG3','TG4','TG14','TG30','TN','TN1','TN2','TN3','TN4','TN14','TN30','TX','TX1','TX3','TX4','TX14','TX30',
                                       'UN', 'UN1', 'UN2', 'UN3', 'UN4', 'UN14', 'UN30', 'UG','UG1','UG2','UG3','UG4','UG14','UG30', 'UX3', 'SQ','SQ1','SQ2','SQ3','SQ4','SQ14','SQ30',
                                       'SP', 'SP1', 'SP2', 'SP3', 'SP4', 'SP30','Q','Q1','Q2','Q3','Q4','Q14','Q30','DR2', 'DR14', 'DR30', 'RH30', 'EV24','EV241','EV242','EV243','EV244','EV2414','EV2430'],
                          'huid': ['FHVEC','FHVEC1','FHVEC2','FHVEC3','FHVEC4','FHVEC14','FHVEC30','FG', 'FG1', 'FG2', 'FG3', 'FG4', 'FG14', 'FG30', 'FHN','FHN1','FHN2','FHN3','FHN4','FHN14','FHN30',
                                   'FHX', 'FHX1', 'FHX2', 'FHX3','FHX4','FHX14','FHX30', 'FXX', 'FXX2', 'FXX14', 'FXX30', 'TG','TG1','TG2','TG3','TG4','TG14','TG30','TN','TN1','TN2','TN3','TN4','TN14','TN30','TX','TX1','TX2','TX3','TX4','TX14','TX30',
                                   'T10N', 'T10N1', 'T10N4', 'T10N14','SQ','SQ1','SQ2','SQ3','SQ4','SQ14','SQ30', 'SP', 'SP1', 'SP2', 'SP3', 'SP4','SP14', 'SP30','Q','Q1','Q2','Q3','Q4','Q14','Q30',
                                   'DR', 'DR2', 'DR3', 'DR14', 'DR30', 'RH14', 'RH30', 'UG','UG1','UG2','UG3','UG4','UG14','UG30', 'UX', 'UX14', 'UX30', 'UN','UN1','UN2','UN3','UN4','UN14','UN30', 'PN30',
                                   'EV24','EV241','EV242','EV243','EV244','EV2414','EV2430']}
    weather_data_disease = weather_data[weather_parameters[disease]]
    disease_dates = [datetime.strptime(s, '%d-%m-%Y') for s in labels.loc[labels[disease] == 1]['DD-MM-YYYY']]
    classes = []
    for i in weather_data_disease.index:
        if datetime.strptime(i, '%Y-%m-%d') in disease_dates:
            classes.append(1)
        else:
            classes.append(0)
    weather_data_disease['class'] = classes
    return weather_data_disease

def main():
    weather_data = getDatasets.get_weather_df()
    labels = getDatasets.get_labeled_df()

    koliek = createData('koliek', weather_data, labels)
    hoefbevangen = createData('hoefbevangen', weather_data, labels)
    luchtweg = createData('luchtweg', weather_data, labels)
    huid = createData('huid', weather_data, labels)

    return koliek, hoefbevangen, luchtweg, huid
