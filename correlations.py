from scipy import stats
import pandas as pd
import getDatasets
from datetime import datetime
import numpy as np
from collections import Counter
import statsmodels.api as sm
from matplotlib import pyplot as plt
from permute.core import two_sample

def prep(df_weather, df_horse, disease):
    disease_dates = [datetime.strptime(s, '%d-%m-%Y') for s in df_horse.loc[df_horse[disease] == 1]['DD-MM-YYYY']]
    dates_counter = Counter(disease_dates)

    a = []
    bs = []
    #disease_df = pd.DataFrame(columns=df_weather.columns)
    disease_df = pd.DataFrame(columns=df_weather.columns)
    non_disease_df = pd.DataFrame(columns=df_weather.columns)

    for i, r in df_weather.iterrows():
        dates_w = datetime.strptime(i, '%Y-%m-%d')
        if dates_w in disease_dates:
            for j in range(dates_counter[dates_w]):
                disease_df = disease_df.append(r, ignore_index=True)
                a.append(1)
                bs.append(r)
        else:
            non_disease_df = non_disease_df.append(r, ignore_index=True)
            a.append(0)
            bs.append(r)
    return non_disease_df, disease_df, a, bs

def prep_v2(df_weather, df_horse, disease, weather):
    disease_dates = [datetime.strptime(s, '%d-%m-%Y') for s in df_horse.loc[df_horse[disease] == 1]['DD-MM-YYYY']]
    dates_counter = Counter(disease_dates)

    a = []
    b = []

    for i, r in df_weather.iterrows():
        dates_w = datetime.strptime(i, '%Y-%m-%d')
        if dates_w in disease_dates:
            for j in range(dates_counter[dates_w]):
                a.append(1)
                b.append(r[weather])
        else:
            a.append(0)
            b.append(r[weather])
    return  a, b


def ttest(disease_df, non_disease_df, weather_variables, disease):
    print(disease)
    weather_dict = {}
    for variable in weather_variables:
        x = list(disease_df[variable])
        y = list(non_disease_df[variable])
        p, t = two_sample(x, y, stat='mean', alternative='two-sided', seed=20, reps=1000)
        weather_dict[variable] = [p,t]
    result_df = pd.DataFrame(data=weather_dict)
    df = result_df.T
    df.columns = ['p-value', 't-value']
    df.to_excel('datasets/t-test' + disease + '_new.xlsx')
    return df


def FDR_controling_fig(disease_df, non_disease_df, weathervalues, disease):
    ttest_df = ttest(disease_df, non_disease_df, weathervalues, disease)
    list = [(i, r['p-value']) for i, r in ttest_df.iterrows()]
    list.sort(key= lambda x: x[1])
    p_values = [p for value, p in list]
    values = [value for value, p in list]
    I = range(1, len(p_values)+1)
    v = len(I)
    q = 0.1
    r = 0
    for i in I:
        if p_values[i-1] <= (i * q)/v:
            r = i
    alpha = (r * q)/v
    print(alpha)

    i_plot = [i / v for i in I]
    i1 = i_plot[:r]
    value_cor = values[:r]
    print(value_cor)
    i2 = i_plot[r:]
    print(p_values)
    p_values1 = p_values[:r]
    p_values2 = p_values[r:]
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.xlabel('i/V')
    plt.ylabel('P-value')
    plt.scatter(i1, p_values1, c = 'blue', alpha = 0.3 )
    plt.scatter(i2, p_values2, c = 'red', alpha = 0.3)
    plt.plot([0, 1], [0, q])
    plt.title(disease, fontsize=30)
    plt.show()


def latex_print_df(df):
    print('\\begin{tables}[]')
    print('\\centering')
    print('\\begin{tabular}{|l||r|r||l||r|r|}\\hline')
    print('& T-value & P-value && T-value & P-value \\\hline\hline')
    for i, r in df.iterrows():
        t, p = r[0]
        print(i + ' & ' + str(np.round(t, 5)) + ' & ' + str(np.round(p, 5)))


def pointbiserial(a, bs, weather_var):
    weather_dict = {}
    for variable in weather_var:
        b = [j[variable] for j in bs]
        r, p = stats.pointbiserialr(a, b)
        weather_dict[variable] = [p, r]
    result_df = pd.DataFrame.from_dict(weather_dict, orient='index', columns = ['p-value', 'r-value'])
    return result_df

def plotQQ(title, data):
    fig = sm.qqplot(np.array(data), line='s')
    f = 20
    fig.suptitle(title, fontsize=f)

def relationship(title, a, b):
    plt.scatter(a, b, alpha=0.3)
    disease = [b[i] for i, j in enumerate(a) if j == 1]
    non_disease = [b[i] for i, j in enumerate(a) if j == 0]
    plt.plot( [0,1], [np.mean(non_disease), np.mean(disease)])
    plt.xticks([0,1])
    plt.title(title)
    plt.show()



def main():
    df_weather = getDatasets.get_weather_df()
    df_horse = getDatasets.get_labeled_df()
    weather_var = list(df_weather.columns)
    dict = {}

    for ziekte, disease in zip(['koliek', 'hoefbevangen', 'luchtweg', 'huid'], ['Colic', 'Laminitis', 'Respiratory', 'Skin']):
        non_disease_df, disease_df, a, bs = prep(df_weather, df_horse, ziekte)
        ttest_df = ttest(disease_df, non_disease_df, weather_var, disease)
        # FDR_controling_fig(disease_df, non_disease_df, weather_var, disease)

        # for variable in weather_var:
            # b = [j[variable] for j in bs]
            # relationship(disease + ' ' + variable, a, b)

    # To get the QQ-plots
    # for i, w in enumerate(['DDVEC', 'FHVEC', 'FG', 'FHX', 'FHN', 'FXX', 'TG', 'TN', 'TX', 'T10N', 'SQ', 'SP', 'Q', 'DR',
    #                        'RH', 'RHX', 'UG', 'UX', 'UN', 'EV24', 'PG', 'PX', 'PN',]):
    #     for j, n in enumerate(['','14','30']):
    #         ax = plt.subplot2grid((7, 34), (j, i))
    #         weather = w+n
    #         weather_data = df_weather[weather]
    #         print(len(weather_data))
    #         wvalue, pvalue = stats.shapiro(weather_data)
    #         plotQQ(weather + " W=" + str(round(wvalue, 5)) + ', pvalue=' + str(round(pvalue, 5)), weather_data)
    #         plt.savefig('figure/QQ_' + weather + ".png")


main()