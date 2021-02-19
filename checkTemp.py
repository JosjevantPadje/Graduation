import pandas as pd

temp_df = pd.read_excel('datasets/labeled.xlsx', encoding='ISO-8859-1', na_filter=False, sheet_name='temp', index_col=0)
r = temp_df['real']
r_new = []
t_new = []
for i in r:
    i = i.replace('[', '')
    i = i.replace(']', '')
    i = i.replace(' ', '')
    i = i.split(',')
    k = []
    for j in i:
        if j:
            k.append(float(j))
    r_new.append(k)
t = temp_df['Temp']
for i in t:
    i = i.replace('[', '')
    i = i.replace(']', '')
    i = i.replace(' ', '')
    i = i.split(',')
    k = []
    for j in i:
        if j and j not in ['42.4', '42.35', '42.33', '42.0', '41.8', '40.05', '39.95', '39.94', '39.12', '38.12','40.0']:
            k.append(float(j))
    t_new.append(k)
d_correct = dict()
d_incorrect = dict()
conf_mat = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
for i, j in zip(r_new, t_new):
    conf_mat[len(j)][len(i)] +=1
    for k in j:
        if k in d_correct:
            d_correct[k][2] += 1
        else:
            d_correct[k] = [0, 0, 1, 0]
        if k in i:
            d_correct[k][1] +=1
        else:
            d_correct[k][0] +=1
    for k in i:
        if k in d_correct:
            d_correct[k][3] += 1
        else:
            d_correct[k] = [0, 0, 0, 1]

print(conf_mat)
df_temp_inc_cor_pred_act = pd.DataFrame.from_dict(d_correct).T
df_temp_inc_cor_pred_act.to_csv('datasets/temp_results.csv')
print(df_temp_inc_cor_pred_act)
#print(temp_df.head())
