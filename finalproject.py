# Final Project
# PPHA 30536 - Data & Programming II
# Boni Oloff Nugraha (12242831) & Harry Satriyo Hendharto (12244891)

# Note: This project may require latest development version of GeoPandas:
# pip install git+git://github.com/geopandas/geopandas.git


import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import wbdata
import datetime
import time
from os.path import exists
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import metrics
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

"""
General grading comments:
- Either write docstrings that add more information than the name of the function, or just omit the docstring
- If you need to define globals, they should be static, and defined at the top wherever possible, not in a function using the global statement
- I liked your small flow charts to describe the code in your writeup
- This is nicely generalized.  Hopefully it's a project that you can use in the future.
"""
def wb_get_dataset_id(dataset_name):
    """Get World Bank (WB) Global Financial Inclusion (Findex) dataset_id"""
    url = 'http://api.worldbank.org/v2/sources?per_page=100&format=json'
    metadata = requests.get(url).json()
    dataset_id = ''
    for indicator in metadata[1]:
        if dataset_name == indicator['name'].lower():
            dataset_id = indicator['id']
    return dataset_id


def wb_get_indicators_id(dataset_id):
    """get WB Findex indicators_id & indicators_name""" #JL: your docstring could be more descriptive, especially for something that appears odd, like turning two Series from a DataFrame into individual objects. once I went through the rest of the code it made sense, but it could have all been here
    url_dataset = 'http://api.worldbank.org/v2/indicator?format=json'
    url_dataset_id = url_dataset + '&source=' + dataset_id
    json_dataset = requests.get(url_dataset_id).json()
    df_indicators = pd.DataFrame(json_dataset[1])
    indicators_id, indicators_name = df_indicators['id'], df_indicators['name']
    return indicators_id, indicators_name


def wb_get_country_id():
    """get countries information (id, name, income_group, etc)"""
    url_country = 'http://api.worldbank.org/v2/country?per_page=500&format=json&source=6'
    json_dataset = requests.get(url_country).json()
    global countries
    countries = pd.DataFrame(json_dataset[1])
    countries.columns = [i.lower() for i in countries.columns]
    countries = countries[countries['capitalcity'].astype(bool)]
    return countries


def wb_get_indicators_data(indicators_id, indicators_name):
    """Get WB Findex indicators"""
    indicators_id_name = dict(zip(indicators_id, indicators_name))
    df_findex = wbdata.get_dataframe(indicators_id_name)
    df_findex = df_findex.reset_index()
    return df_findex


def get_gdp():
    gdp_data = wbdata.get_data('NY.GDP.PCAP.PP.KD',
                               data_date=(datetime.datetime(2017, 1, 1)))
    gdp_data = pd.DataFrame(
        [[i['country']['value'], i['value']] for i in gdp_data],
        columns=['country_name', 'GDP per Capita'])
    return gdp_data


def wb_load():
    """Execute all WB functions"""
    if not exists("findex_data.csv"):
        findex_name = 'global financial inclusion'
        findex_id = wb_get_dataset_id(findex_name)
        findex_indicators_id, findex_indicators_name = wb_get_indicators_id(findex_id)
        df_findex = pd.DataFrame(wb_get_indicators_data(findex_indicators_id, findex_indicators_name))
        df_findex.to_csv("findex_data.csv", index=False)
    else:
        df_findex = pd.read_csv("findex_data.csv")
    return df_findex


def imf_get_structure_codes(db):
    """Get IMF database structure to construct API request."""
    url = f"http://dataservices.imf.org/REST/SDMX_JSON.svc/DataStructure/{db}"
    stru_json = requests.get(url).json()
    stru_df = pd.DataFrame(stru_json['Structure']['KeyFamilies']['KeyFamily']
                           ['Components']['Dimension'])
    return stru_df


def imf_get_dimension_codes(structure_code):
    """Get additional parameters (frequency, geography) to construct API request."""
    url = f"http://dataservices.imf.org/REST/SDMX_JSON.svc/CodeList/{structure_code}"
    json_data = requests.get(url).json()
    df = pd.DataFrame(json_data['Structure']['CodeLists']['CodeList']['Code'])
    return df


def imf_get_indicators_data(db, start, end, **kwargs): #JL: the use of unpacked kwargs implies that any number of kwargs should work (e.g. with iteration), but your code clearly works with three specific ones only, and so those should be the keywords
    """Get IMF FAS indicators"""
    url = f"http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/{db}/"\
          f"{kwargs['dim1']}.{kwargs['dim2']}.{kwargs['dim3']}.?startPeriod={start}&endPeriod={end}"
    data = requests.get(url).json()
    return data


def imf_search_indicators(fas_indicators):
    """Filter IMF FAS indicator of interest"""
    # https://stackoverflow.com/questions/26577516
    searchfor = ['Automated Teller Machines',
                 'Other Depository Corporations, Commercial banks, Number of',
                 'Geographical Outreach, Number of Institutions, Other Depository Corporations, Commercial banks, Number of',
                 'Use of Financial Services, Assets: Outstanding Loans, Commercial banks, Domestic Currency',
                 'Use of Financial Services, Liabilities: Outstanding Deposits, Commercial banks, Domestic Currency']

    # https://stackoverflow.com/questions/38231591
    global fas_indicators_desc
    fas_indicators_desc = pd.concat([fas_indicators.drop(['Description'], axis=1),
                                    fas_indicators['Description'].apply(pd.Series)], axis=1)
    fas_indicators_desc = fas_indicators_desc.drop(columns='@xml:lang')
    fas_indicators_desc = fas_indicators_desc.rename(columns={'@value': 'indicator_id', '#text': 'description'})
    fas_query_indicator_id = fas_indicators_desc[
        fas_indicators_desc['description'].str.contains('|'.join(searchfor), case=False)]
    indicator_array = np.array(fas_query_indicator_id["indicator_id"])
    return indicator_array


def imf_load():
    """Execute all IMF functions"""
    if not exists("fas_data.csv"):
        print("\nStart JSON requests to International Monetary Fund.")

        fas_country = imf_get_dimension_codes("CL_AREA_FAS")
        fas_indicators = imf_get_dimension_codes("CL_INDICATOR_FAS")

        # Get FAS data using for loop because limitation of the API throttle
        country_array = np.array_split(np.array(fas_country['@value']), 25)
        indicator_array = imf_search_indicators(fas_indicators)

        fas_data_list = []
        for country in tqdm(country_array, desc="Loading..."):
            dim1 = "A"
            dim2 = "+".join(country)
            dim3 = "+".join(indicator_array)
            fas_interest_json = imf_get_indicators_data("FAS", 2017, 2018, dim1=dim1, dim2=dim2, dim3=dim3)
            for i in fas_interest_json['CompactData']['DataSet']['Series']: #JL: generally, i and j should only be used for 0-n valued iterators. otherwise, the name of the iterator should be descriptive
                if 'Obs' in i:
                    for j in i['Obs']:
                        if ('@OBS_VALUE' in j) and (type(j) is dict):
                            fas_data_list.append(
                                [i['@REF_AREA'], i['@INDICATOR'],
                                 j['@TIME_PERIOD'], j['@OBS_VALUE']])
            time.sleep(0.75)
        print("JSON requests to International Monetary Fund finished.")

        # Add FAS indicators description and reshaping
        fas_indicators = fas_indicators_desc.rename(columns={'indicator_id': 'Code'})

        df_fas = pd.DataFrame(fas_data_list, columns=['Country', 'Code', 'Year', 'Value'])
        df_fas = df_fas.merge(fas_indicators, on='Code', how='left').drop(columns=['Code'])
        df_fas = df_fas.pivot_table(index=['Country', 'Year'], columns='description',
                                    values='Value', aggfunc=np.sum)
        df_fas = df_fas.reset_index()
        df_fas.to_csv(f"{os.getcwd()}/fas_data.csv", index=False)
        return pd.read_csv("fas_data.csv")
    else:
        return pd.read_csv("fas_data.csv")


def get_clean_df():
    """Combine and clean FAS from IMF and FINDEX from WB"""
    global df_imfwb, countries
    countries = wb_get_country_id()
    if not exists("clean_wb_imf.csv"):
        df_findex = wb_load()
        df_fas = imf_load()
        # Filter for the same country list
        country_list = countries[['iso2code', 'name']]
        country_list = country_list.rename(
            columns={'name': 'country_name', 'iso2code': 'country_iso2code'})
        df_findex = df_findex.rename(columns={'country': 'country_name'})
        df_findex = df_findex.merge(country_list, on='country_name', how='inner')
        df_findex = df_findex.drop(columns='country_iso2code')
        df_fas = df_fas.rename(columns={'Country': 'country_iso2code'})
        df_fas = df_fas.merge(country_list, on='country_iso2code', how='inner')
        df_fas = df_fas.drop(columns='country_iso2code')
        # Clean both WB & IMF dataframe to filter the same year of '2017'
        df_findex = df_findex[df_findex['date'].astype("int") == 2017]
        df_fas = df_fas[df_fas['Year'].astype("int") == 2017]
        # Drop missing values (WB)
        df_findex = df_findex.dropna(thresh=3)
        df_findex = df_findex.dropna(thresh=1, axis=1)
        # Merge WB & IMF dataframes
        df_imfwb = df_findex.merge(df_fas, on='country_name', how='inner')
        # Drop missing values from merged dataframe
        df_imfwb.dropna(thresh=3)
        df_imfwb.dropna(thresh=1, axis=1)
        # Merge WBIMF with GDP per cap
        gdp_data = get_gdp()
        df_imfwb = df_imfwb.merge(gdp_data, on='country_name', how='inner')
        # drop missing values GDP per Cap
        df_imfwb = df_imfwb[df_imfwb['GDP per Capita'].notna()]
        df_imfwb.to_csv("clean_wb_imf.csv")
        return pd.read_csv("clean_wb_imf.csv")
    else:
        return pd.read_csv("clean_wb_imf.csv")


def prepare_data():
    """ Construct features from 6 common groups, transform to log, and replace NA with mean """
    global indicator_group, df_features
    indicator_group = ['Account', 'Financial institution account',
                       'No account', 'Used the internet',
                       'Geographical Outreach', 'Use of Financial Services']
    df_imfwb = get_clean_df()
    y = np.log(df_imfwb['GDP per Capita'])
    df_imfwb = df_imfwb.drop(['country_name', 'GDP per Capita', 'Year', 'Unnamed: 0'], axis=1)
    important_features = []
    for i in indicator_group:
        df_ = df_imfwb.iloc[:, df_imfwb.columns.str.contains(i)]
        info_ = df_.describe().T
        max_median = info_['50%'].max()
        max_median_feat = info_[info_['50%'] == max_median].index[0]
        important_features.append(max_median_feat)
    X = df_imfwb[important_features]
    # Fill NA with mean
    for c in X.columns:
        X[c].fillna(np.mean(X[c]), inplace=True)
    # Transform level to log
    for c in X.columns:
        if '%' not in str(c):
            X[c] = [np.log(x) for x in X[c].tolist()]

    df_imfwb = get_clean_df()
    X.columns = indicator_group
    df = pd.concat([X, df_imfwb['country_name']], axis=1, sort=False)
    # add income_group category
    income_group = pd.concat([countries[['id', 'name', 'incomelevel']],
                             countries['incomelevel'].apply(pd.Series)], axis=1)
    income_group = income_group.drop(columns=['id', 'iso2code'])
    income_group = income_group.rename(columns={'name': 'country_name', 'value': 'income_group'})
    df_features = df.merge(income_group, on='country_name', how='left')
    df_features = df_features.merge(df_imfwb[['country_name', 'GDP per Capita']], on='country_name', how='left')
    df_features['log(Geo Outreach)'] = np.log(df_features['Geographical Outreach'])
    df_features['log(Use of Financial Services)'] = np.log(df_features['Use of Financial Services'])
    df_features['log(GDP per Capita)'] = np.log(df_features['GDP per Capita'])
    df_features = df_features.drop(columns=['Geographical Outreach', 'Use of Financial Services',
                                            'GDP per Capita', 'incomelevel'])
    return X, y, df_features


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_all_model():
    X, y, _ = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    results = [('Model', 'MAE', 'MSE', 'RMSE')]

    for name, model in mod_dic.items():
        pred = fit_predict(model, X_train, y_train, X_test)
        mae = round(metrics.mean_absolute_error(y_test, pred), 2)
        mse = round(metrics.mean_squared_error(y_test, pred), 2)
        rmse = round(np.sqrt(metrics.mean_squared_error(y_test, pred)), 2)
        results.append((name, mae, mse, rmse))
        # plotting
        sns.regplot(x=y_test, y=pred) #JL: plotting should be done on an axis object, and not on plt (like you did in the boxplot function)
        plt.title(f'{name}')
        plt.ylabel('Prediction GDP')
        plt.xlabel('True GDP')
        plt.tight_layout()
        plt.savefig(f'{name} graph.png', dpi=300)
        plt.show()

    # Find the coefficients of Linear Regression
    mod = LinearRegression()
    mod.fit(X_train, y_train)
    coef = pd.DataFrame({'Variable': X_train.columns.tolist(), 'Coef': mod.coef_})

    print('\n Prediction Performance Result')
    for line in results:
        print(line[0].ljust(45), str(line[1]).ljust(20), str(line[2]))

    print(f'\n Coefficients of Linear Regression:')
    print(coef)


def run_cross_validation():
    X, y, _ = prepare_data()

    global mod_dic
    mod_dic = {'Linear Regression': LinearRegression(),
               'Stochastic Gradient Descent Regression':
                   make_pipeline(StandardScaler(), SGDRegressor(loss='squared_loss', penalty='l2')),
               'Polynomial Regression': Pipeline(
                   [('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=True))])
               }

    scoring = ['r2', 'neg_mean_squared_error']
    results = [('Model', 'Negative MSE', 'R-squared')]

    for n, m in mod_dic.items():
        scores = cross_validate(m, X, y, scoring=scoring, cv=5)
        mse = round(np.mean(scores['test_neg_mean_squared_error']), 2)
        r2 = round(np.mean(scores['test_r2']), 2)
        results.append((n, mse, r2))

    print('\n Cross Validation Result')
    for line in results:
        print(line[0].ljust(45), str(line[1]).ljust(20), str(line[2]))


def add_geom(df_imfwb, countries):
    geom = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    geom = geom.rename(columns={'iso_a3': 'id'})
    df_imfwb = df_imfwb.rename(columns={'country_name': 'name'})
    df_imfwb = df_imfwb.merge(countries[['name', 'id']], on='name', how='left')
    df_imfwb.drop('name', axis=1)
    imfwb_geom = geom.merge(df_imfwb, on=['id'], how='left')
    return imfwb_geom


def plot_worldmap(df_geom, indicator_to_map):
    # https://geopandas.org/mapping.html
    df_geom.rename(columns={ df_geom.columns[-5]: 'Number of Bank Branches' }, inplace=True)
    indicator_to_map = ['Account (% age 15+)', 'Number of Bank Branches','GDP per Capita']
    dataset_title = ['World Bank Findex', 'IMF FAS', 'World Development Indicators']
    color_set = ['Blues', 'Greens', 'Reds']

    for indicator, dataset, color in zip(indicator_to_map, dataset_title, color_set):
        _, ax = plt.subplots(figsize=(15, 15))
        plt.title(dataset + ' : ' + indicator, fontsize=20, color='black')
        ax.axis('off')
        ax.annotate('Source: WB Findex & IMF FAS, 2017', xy=(-180, -55), fontsize=10, color='#555555')
        ax.annotate('*) Grey area represents non-participant countries both for Findex and FAS Survey',
            xy=(-180, -60), fontsize=9, color='#555555')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        if '%' in indicator:
            norm = None
        else:
            norm = colors.LogNorm(vmin=df_geom.iloc[:, -1].min(), vmax=df_geom.iloc[:, -1].max())
            ax.annotate('(log)', xy=(195, 90), fontsize=10, color='black')
        df_geom[df_geom.continent != 'Antarctica'].plot(
            column=indicator, cmap=color, ax=ax, edgecolor='black', linewidth=0.3, legend=True, cax=cax,
            missing_kwds=dict(color="darkgrey"), norm=norm)
        plt.tight_layout()
        plt.savefig(f'Map of {indicator}.png')
        plt.show()


def boxplot(df_imfwb):
    for i in indicator_group:
        ind_group = df_imfwb.filter(regex=i)
        ind_group.columns.name = 'indicator_name'
        ind_group = pd.DataFrame(ind_group.stack())
        ind_group.columns = ['value']
        ind_group = ind_group.reset_index()
        ind_group = ind_group.drop(columns='level_0')

        _, ax = plt.subplots(figsize=(15, 7.5)) #JL: the first value is the figure object, which by convention is usually saved as "fig". it can be particularly useful with subplots
        order = ind_group.groupby(by=["indicator_name"])["value"].median().sort_values().iloc[::-1].index
        ax = sns.boxplot(x=ind_group['value'], y=ind_group['indicator_name'], order=order,
                         palette='Blues_r').set_title('Summary Statistics of : ' + i)
        ax = sns.stripplot(x="value", y="indicator_name", data=ind_group,
                           order=order, jitter=True, size=2, color=".4", linewidth=0)
        ax.set(xlabel=" ", ylabel=" ")
        if '%' not in ' '.join(df_imfwb.filter(regex=i).columns):
            ax.set(xscale="log")
        plt.tight_layout()
        plt.savefig(f'Summary {i}.png', dpi=300)
        plt.show()


def pair_plot(df_imfwb, countries):
    _, _, df_features = prepare_data()
    # plotting
    g = sns.pairplot(df_features, hue='income_group', diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'})
    g._legend.remove()
    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g.fig.legend(handles=handles, labels=labels, loc='upper right', ncol=4)

    plt.suptitle('Pair plot of 6 main features + 1 label', x=0.1, y=1.05, fontsize=20)
    plt.tight_layout()
    plt.savefig('PairPlot.png', dpi=300)
    plt.show()


def summary_stat():
    X, y, _ = prepare_data()
    X['gdppercap'] = y
    print('Summary Statistics')
    print(X.describe().T)


def plotting(df_imfwb):
    indicator_to_map = 'Account (% age 15+)'
    imfwb_geom = add_geom(df_imfwb, countries)
    plot_worldmap(imfwb_geom, indicator_to_map)
    boxplot(df_imfwb)
    pair_plot(df_imfwb, countries)


def run_plot_pca(df_imfwb):
    X, y, df_features = prepare_data()
    df_pca = pd.concat([X, df_features], axis=1, sort=False)
    X_train, _, _, _ = train_test_split(df_pca, y, test_size=0.2)
    pca = PCA(n_components=2)
    pca.fit(X_train.iloc[:, 0:6])
    X_train_pca = pca.transform(X_train.iloc[:, 0:6])
    X_train['PCA1'] = X_train_pca[:, 0]
    X_train['PCA2'] = X_train_pca[:, 1]
    sns.lmplot('PCA1', 'PCA2', hue='income_group', data=X_train, fit_reg=False, legend=False)
    plt.title('PCA Analysis')
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig('PCA.png', dpi=300)
    plt.show()


def main():
    df_imfwb = get_clean_df()
    summary_stat()
    plotting(df_imfwb)
    run_plot_pca(df_imfwb)
    run_cross_validation()
    run_all_model()


main()
