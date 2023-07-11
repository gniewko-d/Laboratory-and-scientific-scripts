# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:03:23 2023

@author: gniew
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import sem
import pingouin as pg
from scipy import stats
import scikit_posthocs as sp
import seaborn as sns
noramlizer = MinMaxScaler()


pathway = "G:/NI_VTA/Anatomia/Tracing/Anterograde_włókna/Area_fraction.xlsx"
#pathway = "C:/Users/gniew/Desktop/PRACA/doktorat/Area_fraction.xlsx"
range_in_dv = [i for i in range(6, 19)]
degrees = np.arange(1, 10)
min_rmse, min_deg = 1e10, 0
rmses = []

df = pd.read_excel(pathway)
#df.iloc[range_in_dv, 0:17] = noramlizer.fit_transform(df.iloc[range_in_dv, 0:17])
df_1 = pd.read_excel(pathway)
column_title = df.columns.tolist()
result_mean = pd.DataFrame(0, index=range(0,13), columns = ["rostral_mean", "central_mean", "caudal_mean", "DV"])
result_median = pd.DataFrame(0, index=range(0,13), columns = ["rostral_median", "central_median", "caudal_median", "DV"])

df_modeling = pd.DataFrame(columns = ["Area_fraction", "DV", "Group"])
group = ["rostral_mean", "central_mean", "caudal_mean"]
group_median = ["rostral_median", "central_median", "caudal_median"]
group_v1 = ["Rostral", "Central", "Caudal"]
rostral_index= [column_title.index(i) for i in column_title if "rostral" in i]
central_index= [column_title.index(i) for i in column_title if "central" in i]
caudal_index= [column_title.index(i) for i in column_title if "caudal" in i]
result_mean.loc[:, "DV"] = df.iloc[range_in_dv, 17].tolist()
result_median.loc[:, "DV"] = df.iloc[range_in_dv, 17].tolist()
list_container = [rostral_index, central_index, caudal_index]


for j,i in enumerate(list_container):
    mean_dv = df.iloc[range_in_dv, i].mean(axis = 1).tolist()
    median_dv = df.iloc[range_in_dv, i].median(axis = 1).tolist()
    ans1 = [df.iloc[range_in_dv, ii].tolist() for ii in i]
    ans1 = [item for sublist in ans1 for item in sublist]
    ans2 = df.iloc[range_in_dv, 17].tolist()* len(i)
    ans3 = [group[j] for kk in range(len(ans2))]
    dict_to_append = {"Area_fraction" : ans1, "DV" : ans2, "Group": ans3}
    df_modeling = df_modeling.append(pd.DataFrame(dict_to_append))
    result_mean.iloc[:, j] = mean_dv
    result_median.iloc[:, j] = median_dv

dv_range = np.arange(7.65, 8.95, 0.1).tolist()
dv_range = [round(i,2) for i in dv_range]
y_modeling = df_modeling.reset_index(drop=True)
y_modeling_all = [y_modeling.loc[y_modeling["DV"] == i, "Area_fraction"].mean() for i in dv_range]

list_dv = [7.85, 8.35, 8.85]
box_plot = df_modeling.loc[df_modeling["DV"].isin(list_dv), :]
df_ANOVA = df_modeling.loc[(df_modeling["DV"] == list_dv[0]) | (df_modeling["DV"] == list_dv[1]) | (df_modeling["DV"] == list_dv[2]), :]

for feature in group:
    # Data split
    X = result_mean.loc[:, "DV"].tolist()
    y = result_mean.loc[:, feature].tolist()
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 10)
    for deg in degrees:
    
    # Train features
    
        poly_features = PolynomialFeatures(degree=deg, include_bias=False)
        #x_poly_train = poly_features.fit_transform(np.asarray(x_train).reshape(-1, 1))
        x_poly_train = poly_features.fit_transform(np.asarray(X).reshape(-1, 1))
        # Linear regression
        poly_reg = LinearRegression()
        poly_reg.fit(x_poly_train, y)

    # Compare with test data
        #x_poly_test = poly_features.fit_transform(np.asarray(x_test).reshape(-1, 1))
        poly_predict = poly_reg.predict(x_poly_train)
        poly_mse = mean_squared_error(y, poly_predict)
        poly_rmse = np.sqrt(poly_mse)
        rmses.append(poly_rmse)
    
    # Cross-validation of degree
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg
    print(f'Best degree {min_deg} with RMSE {min_rmse} for feature {feature}')
print("#" * 70)

list_of_degrees = [9,9,9]
list_of_prediction = []

# =============================================================================
# for i, feature in enumerate(group):
#     X = result_mean.loc[:, "DV"].tolist()
#     #y = result_mean.loc[:, feature].tolist()
#     y = y_modeling_all
#     poly = PolynomialFeatures(degree=list_of_degrees[i], include_bias=False)
#     poly_features = poly.fit_transform(np.asarray(X).reshape(-1, 1))
#     poly_reg_model = LinearRegression()
#     poly_reg_model.fit(poly_features, y)
#     y_predicted = poly_reg_model.predict(poly_features)
#     print(f"R2 score for feature {feature} is {round(r2_score(y, y_predicted),2)}")
#     list_of_prediction.append(y_predicted.tolist())
# =============================================================================
y_modeling_all_log = np.log(y_modeling_all)
curve_fit = np.polyfit(dv_range,y_modeling_all_log, 1)
y_exp = np.exp(curve_fit[1]) * np.exp(curve_fit[0]*np.array(dv_range))

fig, axs = plt.subplots(3, figsize=(7, 20))
fig.tight_layout(pad=10.0)
print(f"R2 score for is {round(r2_score(y_modeling_all, y_exp),2)}")
for j,i in enumerate(group):
    #axs[j].scatter(result_mean.loc[:, "DV"].tolist(), result_mean.loc[:, i].tolist())
    axs[j].scatter(result_mean.loc[:, "DV"].tolist(), y_modeling_all)
    #axs[j].plot(result_mean.loc[:, "DV"].tolist(), list_of_prediction[j], color = "r", label= 'Polynomial regression line')
    axs[j].plot(result_mean.loc[:, "DV"].tolist(), y_exp, color = "r", label= 'exponential curve')
    axs[j].legend()
    axs[j].set_title(group_v1[j] + " part of VTA")
    axs[j].set_xlabel("DV axis [mm]")
    axs[j].set_ylabel("Normalized area fraction")
save_fig_to = "C:/Users/gniew/OneDrive/Pulpit/tracing_python/figs/VTA_exp_curve.svg"
plt.savefig(save_fig_to)
plt.show()
# Normal test
for i in group:
    ap_R = [print(pg.normality(df_ANOVA.loc[(df_ANOVA["DV"] == kk) & (df_ANOVA["Group"] == i), "Area_fraction"])) for kk in list_dv]

# Kruskal

# DV
print(stats.kruskal(df_ANOVA.loc[df_ANOVA["DV"] == list_dv[0], "Area_fraction"].tolist(), df_ANOVA.loc[df_ANOVA["DV"] == list_dv[1], "Area_fraction"].tolist(), df_ANOVA.loc[df_ANOVA["DV"] == list_dv[2], "Area_fraction"].tolist()))

post_hoc_area = [df_ANOVA.loc[df_ANOVA["DV"] == list_dv[0], "Area_fraction"].tolist(), df_ANOVA.loc[df_ANOVA["DV"] == list_dv[1], "Area_fraction"].tolist(), df_ANOVA.loc[df_ANOVA["DV"] == list_dv[2], "Area_fraction"].tolist()]

df_post_hoc_area = sp.posthoc_dunn(post_hoc_area)
df_post_hoc_area.rename(columns = {1: 7.85, 2: 8.35, 3: 8.85}, inplace = True)
df_post_hoc_area.rename(index = {1: 7.85, 2: 8.35, 3: 8.85}, inplace = True)

fig, ax = plt.subplots()
fig.tight_layout()

#dict_for_bar = {"R": result_median.loc[result_median["DV"].isin([list_dv]), "mean"].tolist(), "CE": df_count_cell_mean.loc[df_count_cell_mean["slice"] == "CE", "mean"].tolist(), "CA": df_count_cell_mean.loc[df_count_cell_mean["slice"] == "CA", "mean"].tolist()}
dict_for_bar = {i: result_median.loc[result_median["DV"].isin(list_dv), i].tolist() for i in group_median}


x = np.arange(len(list_dv))
width = 0.15
multiplier = 0
fig, ax = plt.subplots()

#sem_list = [[6.2, 5.51], [10.07, 9.98], [12.86, 6.02]]
#for key, value in dict_for_bar.items():
#    print(value)
#    offset = width * multiplier
#    rects = ax.bar(x + offset, value, width, label=key)
    #ax.bar_label(rects, padding=3)
#    multiplier += 1
#ax.set_ylim(0, 8)
#ax.set_xticks(x + width, list_dv)
#ax.legend(loc='upper left')
save_fig_to = "C:/Users/gniew/OneDrive/Pulpit/tracing_python/figs/VTA_boxplot_antero.svg"
#plt.savefig(save_fig_to)
#plt.show()

sns.set_theme("notebook")
sns.boxplot(data = box_plot, x = "DV", y = "Area_fraction", hue = "Group")
plt.savefig(save_fig_to)



#df_post_hoc_area = df_post_hoc_area.round(5)

# group
#print(stats.kruskal(df_ANOVA.loc[df_ANOVA["Group"] == group[0], "Area_fraction"].tolist(), df_ANOVA.loc[df_ANOVA["Group"] == group[1] & (df_ANOVA["DV"] == 7.85), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[2]) & (df_ANOVA["DV"] == 7.85), "Area_fraction"].tolist()))
print(stats.kruskal(df_ANOVA.loc[(df_ANOVA["Group"] == group[0]) & (df_ANOVA["DV"] == 7.85), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[1]) & (df_ANOVA["DV"] == 7.85), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[2]) & (df_ANOVA["DV"] == 7.85), "Area_fraction"].tolist()))
print(stats.kruskal(df_ANOVA.loc[(df_ANOVA["Group"] == group[0]) & (df_ANOVA["DV"] == 8.35), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[1]) & (df_ANOVA["DV"] == 8.35), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[2]) & (df_ANOVA["DV"] == 8.35), "Area_fraction"].tolist()))
post_hoc_8_35 = [df_ANOVA.loc[(df_ANOVA["Group"] == group[0]) & (df_ANOVA["DV"] == 8.35), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[1]) & (df_ANOVA["DV"] == 8.35), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[2]) & (df_ANOVA["DV"] == 8.35), "Area_fraction"].tolist()]
df_post_hoc_8_35 = sp.posthoc_dunn(post_hoc_8_35)

print(stats.kruskal(df_ANOVA.loc[(df_ANOVA["Group"] == group[0]) & (df_ANOVA["DV"] == 8.85), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[1]) & (df_ANOVA["DV"] == 8.85), "Area_fraction"].tolist(), df_ANOVA.loc[(df_ANOVA["Group"] == group[2]) & (df_ANOVA["DV"] == 8.85), "Area_fraction"].tolist()))

