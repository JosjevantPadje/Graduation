import getDatasets
from datetime import datetime
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import calendar
import pylab as P
import seaborn as sns
from scipy.stats import binned_statistic
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

def split_tup_list(tl):
    xs, ys = [], []
    for x, y in tl:
        xs.append(x)
        ys.append(y)
    return xs, ys

def prepare_for_plotting(all, disease, normalize):
    c_all = dict(Counter(all))
    c_disease = dict(Counter(disease))
    result_dict = dict()

    if normalize:
        for y, v in c_all.items():
            if y in c_disease.keys():
                result_dict[y] = float(c_disease[y]*100/float(v))
            else:
                result_dict[y] = 0
    else:
        for y, v in c_all.items():
            if y in c_disease.keys():
                result_dict[y] = c_disease[y]
            else:
                result_dict[y] = 0

    xy = list(result_dict.items())
    xy.sort(key=lambda tup: tup[0])
    xs, ys = split_tup_list(xy)
    objects = xs
    y_pos = np.arange(len(objects))
    performance = ys
    return y_pos, performance, objects

def get_year_month_visualization(consults_df, disease, title):
    years_all = []
    months_all = []
    years_disease = []
    months_disease = []

    for i, r in consults_df.iterrows():
        years_all.append(datetime.strptime(r['DD-MM-YYYY'], '%d-%m-%Y').year)
        months_all.append(datetime.strptime(r['DD-MM-YYYY'], '%d-%m-%Y').month)
        if r[disease] == 1:
            years_disease.append(datetime.strptime(r['DD-MM-YYYY'], '%d-%m-%Y').year)
            months_disease.append(datetime.strptime(r['DD-MM-YYYY'], '%d-%m-%Y').month)
    y_pos_year, performance_year, objects_year = prepare_for_plotting(years_all, years_disease, True)
    y_pos_month, performance_month, objects_month = prepare_for_plotting(months_all, months_disease, False)
    plt_year = consults_per_year(y_pos_year, performance_year, objects_year, title)
    plt_year.savefig('figure/' + disease + '_per_year.JPG')
    plt_year.show()
    plt_month = consults_per_month(y_pos_month, performance_month, title)
    plt_month.savefig('figure/' + disease + '_per_month.JPG')
    plt_month.show()

def consults_per_year(y_pos, performance, objects, title):
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation=-40)
    plt.ylabel('% of consults that concern ' + title)
    plt.xlabel('Year')
    return plt


def consults_per_month(y_pos, performance, title):
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(np.arange(12), calendar.month_name[1:13], rotation=-40)
    plt.ylabel('Count of consults that concern ' + title)
    plt.xlabel('Month')
    return plt

def disease_weather(ax, disease_dates, weather_df, weather_x, weather_y, title, name, ylabel):
    disease_results_x = []
    non_disease_results_x = []
    disease_results_y = []
    non_disease_results_y = []
    # hue = []

    for i, r in weather_df.iterrows():
        if datetime.strptime(i, '%Y-%m-%d') in disease_dates:
            disease_results_x.append(r[weather_x])
            disease_results_y.append(r[weather_y])
            # hue.append(title)
        else:
            # hue.append('no ' + title)
            non_disease_results_x.append(r[weather_x])
            non_disease_results_y.append(r[weather_y])
    # new_df = pd.DataFrame()
    # new_df['disease'] = hue
    # new_df['y'] = list(weather_df[weather_y])
    # new_df['d0'] = list(pd.cut(weather_df[weather_x], 5))
    non_zip = random.sample(list(zip(non_disease_results_x, non_disease_results_y)), 100)
    non_sample_x = [x for x, y in non_zip]
    non_sample_y = [y for x, y in non_zip]
    d_zip = random.sample(list(zip(disease_results_x, disease_results_y)), 100)
    d_sample_x = [x for x, y in d_zip]
    d_sample_y = [y for x, y in d_zip]

    ax.scatter(non_sample_x, non_sample_y, c='blue', label='non ' +title, alpha=0.3, edgecolors='none')
    ax.scatter(d_sample_x, d_sample_y, c='red', label=title, alpha=0.3, edgecolors='none')
    f = 20
    if ylabel ==  'y = 30':
        ax.set_xlabel(r'x on $d_{t}$', fontsize=f)
    if weather_x == 'TG':
        ax.set_ylabel(ylabel, fontsize=f)
    if ylabel == r'x on $d_{t-1}$':
        ax.set_title('x = ' + weather_x, fontsize=f)
    ax.tick_params(labelsize = 'large')
    # ax = sns.swarmplot(x="d0", y="y", hue="disease", data=new_df)




def plot_disease_weather(consults_df, weather_df,disease, title):

    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)

    for j, z in enumerate(zip(['TG', 'PG', 'FG', 'DR', 'UG'],
                             ['temperature', 'sea level pressure', 'windspeed', 'precipitation duration',
                              'relative atmospheric humidity'])):
        weather, name = z
        disease_dates = []
        for i, r in consults_df.iterrows():
            if r[disease] == 1 :
                disease_dates.append(datetime.strptime(r['DD-MM-YYYY'], '%d-%m-%Y'))
        l = [r'x on $d_{t-1}$',
             r'x on $d_{t-2}$',
             r'x on $d_{t-3}$',
             r'x on $d_{t-4}$',
             'y = 14',
             'y = 30']

        for k, a in enumerate(zip(['1', '2', '3', '4', '14', '30'], l)):
            i, ylabel = a
            ax = plt.subplot2grid((6, 5), (k, j))
            disease_weather(ax, disease_dates, weather_df, weather, weather+i, title, name, ylabel)

def weather_plots(df, temp):
    date = df.index
    d = dict()
    d_count = dict()
    for da, t in zip(date, temp):
        m = datetime.strptime(da, '%Y-%m-%d').month
        if m in d:
            d[m] += t
            d_count[m] += 1
        else:
            d[m] = t
            d_count[m] = 1

    for key, value in d.items():
        d[key] = value/d_count[key]

    xy = list(d.items())
    xy.sort(key=lambda tup: tup[0])
    xs, ys = split_tup_list(xy)
    objects = xs
    y_pos = np.arange(len(objects))
    performance = ys
    consults_per_month(y_pos, performance, '')


def correlation_weather_disease(disease, weather, df_labeled, df_weather, y_label, x_label):
    weather_values = df_weather[weather]
    disease_dates = [datetime.strptime(s, '%d-%m-%Y') for s in df_labeled.loc[df_labeled[disease] == 1]['DD-MM-YYYY']]


    d = {}

    x_weather = list(weather_values)
    x_disease = []
    for index, value in weather_values.items():
        if value not in d.keys():
            d[value] = 0
        x_weather.append(value)
        date_w = datetime.strptime(index, '%Y-%m-%d')
        for date in disease_dates:
            if date == date_w:
                x_disease.append(value)
                if value in d.keys():
                    d[value] += 1


    P.figure()

    n, bins, patches = P.hist([x_disease, x_weather], 10, histtype='bar',
                              color=['crimson', 'burlywood'],
                              label=['Colic', 'DDVEC'])

    P.show()
    print(bins)
    x = [i/j for i, j in zip(n[0], n[1])]
    y = [int(b) for b in bins]
    y.pop(0)
    objects = y
    y_pos = np.arange(len(objects))
    performance = x

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Normalized ' + y_label)
    plt.xlabel(x_label)

    plt.show()
    return d

def scatter_hist(x, dx, y, dy, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, label='no colic', alpha=0.3)
    ax.scatter(dx, dy, label='colic', alpha=0.3)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist([x, dx], bins=20)#, histtype='step')
    ax_histy.hist([y, dy], bins=20, orientation='horizontal')#, histtype='step')

def swarm():
    consults_df = getDatasets.get_labeled_df()
    weather_df = getDatasets.get_weather_df()

    disease_dates = dict()

    for ziekte, disease in [('koliek', 'colic'), ('hoefbevangen', 'laminitis'), ('luchtweg', 'respiratory'), ('huid', 'skin')]:
        disease_dates[disease] = [datetime.strptime(s, '%d-%m-%Y') for s in
                     consults_df.loc[consults_df[ziekte] == 1]['DD-MM-YYYY']]

    weather_var = {'colic':'PN4', 'laminitis':'TX2', 'respiratory':'UN2', 'skin':'Q2'}
    for disease in ['colic', 'laminitis', 'respiratory', 'skin']:
        disease_dates_count = Counter(disease_dates[disease])
        dy = []
        dhue = []
        nhue = []
        ny = []
        y= []
        hue=[]
        swarm_df = pd.DataFrame()
        weather = weather_var[disease]
        for i, r in weather_df.iterrows():
            dates_w = datetime.strptime(i, '%Y-%m-%d')
            if dates_w in disease_dates[disease]:
                for j in range(disease_dates_count[dates_w]):
                    dy.append(r[weather])
                    dhue.append(disease)
            else:
                nhue.append('no ' + disease)
                ny.append(r[weather])
        ds = random.sample(list(zip(dy, dhue)), np.min([len(dy), 150]))
        ns = random.sample(list(zip(ny, nhue)), np.min([len(ny), 150]))
        y.extend([y for y,_ in ds])
        hue.extend([hue for _,hue in ds])
        y.extend([y for y, _ in ns])
        hue.extend([hue for _, hue in ns])
        swarm_df[weather] = y
        swarm_df['disease'] = hue
        plt.rc('font', size=15)
        plt.rc('axes', titlesize=20)
        fig, ax = plt.subplots()

        #plt.plot([dhue[0], nhue[0]], [np.mean(dy), np.mean(ny)])
        ax = sns.boxplot(x='disease', y=weather, data=swarm_df, whis=np.inf)
        #ax = sns.swarmplot(x=weather, y="y", data=swarm_df)
        ax = sns.swarmplot(x='disease', y=weather, data=swarm_df, color=".2")

        plt.show()


    #
    # fig, ax = plt.subplots()
    # ax = sns.swarmplot(x="x", y="y", hue="hue", data=swarm_df, palette="Set2", dodge=True)
    # plt.show()

swarm()


def main_scatter_hist():
    consults_df = getDatasets.get_labeled_df()
    weather_df = getDatasets.get_weather_df()
    disease_dates = [datetime.strptime(s, '%d-%m-%Y') for s in consults_df.loc[consults_df['hoefbevangen'] == 1]['DD-MM-YYYY']]
    weather_x = 'PX3'
    weather_y = 'TG1'
    d_x = []
    non_x = []
    d_y = []
    non_y = []
    # hue = []

    for i, r in weather_df.iterrows():
        if datetime.strptime(i, '%Y-%m-%d') in disease_dates:
            d_x.append(r[weather_x])
            d_y.append(r[weather_y])
            # hue.append(title)
        else:
            # hue.append('no ' + title)
            non_x.append(r[weather_x])
            non_y.append(r[weather_y])

    print("mean_lam: ", np.mean(d_x))
    print("mean_non: ", np.mean(non_x))
    print('mean_lam - mean_non', np.mean(d_x) - np.mean(non_x))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    plt.rc('font', size=20)

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(non_x, d_x, non_y, d_y, ax, ax_histx, ax_histy)

    ax.set_xlabel(r'TG on $d_{t}$   TG on $d_{t-1}$')
    ax.set_ylabel(r'Average DR over 14 days before $d_{t}$')

    ax.legend()
    plt.show()
# main_scatter_hist()


def main():
    consults_df = getDatasets.get_labeled_df()
    weather_df = getDatasets.get_weather_df()
    # correlation_dict = {'koliek': ['FHVEC30', 'FG30', 'FHX30', 'FHNH30', 'FXX30', 'FXXH30', 'TNH30', 'RH30', 'RHXH30', 'UX30', 'UXH30', 'UN30', 'UNH30', 'PXH30', 'FHVEC14', 'FG14', 'FHX14', 'FHNH14', 'UX14', 'UXH14', 'UN14', 'PXH14', 'UG4', 'UN4', 'UN3', 'PG3', 'PN3', 'PG2', 'PN2'], 'hoefbevangen': ['FG', 'TG', 'TN', 'TX', 'SQ', 'Q', 'UG', 'UX', 'UN', 'EV24', 'FHVEC30', 'FG30', 'FHX30', 'FHN30', 'TG30', 'TX30', 'SQ30', 'SP30', 'Q30', 'DR30', 'RHXH30', 'UG30', 'UX30', 'UXH30', 'UN30', 'UNH30', 'EV2430', 'PXH30', 'FHVEC14', 'FG14', 'FHX14', 'FHN14', 'TG14', 'TX14', 'TXH14', 'SQ14', 'SP14', 'Q14', 'DR14', 'RH14', 'RHXH14', 'UG14', 'UX14', 'UN14', 'UNH14', 'EV2414', 'TG4', 'TX4', 'SQ4', 'Q4', 'UG4', 'UN4', 'EV244', 'FHVEC3', 'FG3', 'FHX3', 'FHN3', 'TG3', 'TX3', 'SQ3', 'SP3', 'Q3', 'UG3', 'UN3', 'EV243', 'FHVEC2', 'FG2', 'TG2', 'TX2', 'SQ2', 'SP2', 'Q2', 'RHXH2', 'UG2', 'UN2', 'EV242', 'FHVEC1', 'FHN1', 'TG1', 'TX1', 'SQ1', 'Q1', 'UG1', 'UN1', 'EV241'], 'luchtweg': ['FHVEC', 'FG', 'FHX', 'FHN', 'TG', 'TX', 'SQ', 'Q', 'UG', 'UN', 'EV24', 'FHVEC30', 'FG30', 'FHX30', 'FHN30', 'TG30', 'TX30', 'T10NH30', 'SQ30', 'SP30', 'Q30', 'DR30', 'RHXH30', 'UG30', 'UXH30', 'UN30', 'UNH30', 'EV2430', 'PXH30', 'FHVEC14', 'FG14', 'FHX14', 'FHN14', 'TG14', 'TN14', 'TX14', 'SQ14', 'SP14', 'Q14', 'DR14', 'RHXH14', 'UG14', 'UN14', 'UNH14', 'EV2414', 'PXH14', 'TG4', 'TN4', 'TX4', 'T10NH4', 'SQ4', 'SP4', 'Q4', 'UG4', 'UN4', 'EV244', 'TG3', 'TX3', 'SQ3', 'SP3', 'Q3', 'UG3', 'UN3', 'EV243', 'FHVEC2', 'FG2', 'FHN2', 'TG2', 'TX2', 'SQ2', 'SP2', 'Q2', 'DR2', 'UG2', 'UN2', 'EV242', 'FHVEC1', 'FG1', 'FHN1', 'TG1', 'TX1', 'SQ1', 'SP1', 'Q1', 'RHXH1', 'UG1', 'UN1', 'EV241'], 'huid': ['FHVEC', 'FG', 'FHN', 'UG', 'UN', 'FHVEC30', 'SP30', 'UG30', 'UXH30', 'UN30', 'UNH30', 'PXH30', 'FHVEC14', 'FHNH14', 'UG14', 'UXH14', 'UN14', 'UG4', 'UN4', 'UG3', 'UN3', 'UG2', 'UN2', 'UG1', 'UN1']}
    for disease, title in zip(['koliek', 'hoefbevangen', 'luchtweg', 'huid'], ['colic', 'laminitis', 'respiratory', 'skin']):
        #get_year_month_visualization(consults_df, disease, title)
        plot_disease_weather(consults_df, weather_df, disease, title)
        plt.show()
        # for var in [weather_df['TG'], weather_df['DR'], weather_df['RH']]:
           # weather_plots(weather_df, var)
           # plt.show()
        # for weather, x in zip(['TG', 'PG', 'FG', 'DR', 'UG'],
        #                       ['temperature', 'sea level pressure', 'windspeed', 'precipitation duration',
        #                        'relative atmospheric humidity']):
        #     correlation_weather_disease(disease, weather, consults_df, weather_df, title, x)

# main()
