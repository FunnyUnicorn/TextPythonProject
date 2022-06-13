import streamlit as st
with st.echo(code_location='below'):
    import pandas as pd
    import plotly.express as px
    import numpy as np
    import geopandas as gpd
    import requests
    from sklearn.linear_model import LinearRegression
    import sqlite3

    name = st.text_input("Name", key="name", value="???")
    st.write(f"### Hello, {name}!")

    st.header('Brief description of the project.')

    st.write('In my final project, I did research on the popularity of certain search queries and calculated some statistical indices for them. I hope you enjoy reading it!')

    st.header('Popularity charts: Russia')

    st.write('Here you can graph the popularity of queries about various famous people and look at the dynamics of this value.')

    def get_data(x):
        data_url = x
        return pd.read_csv(data_url)

    morgen_time = get_data("https://raw.githubusercontent.com/FunnyUnicorn/NewPythonProject/main/MorgensternTimeline4Years.csv").drop(labels = ['Week'],axis = 0).rename(columns={"Category: All categories": "Morgenstern"})
    putin_time = get_data("https://raw.githubusercontent.com/FunnyUnicorn/NewPythonProject/main/PutinTimeline4Years.csv").drop(labels = ['Week'],axis = 0).rename(columns={"Category: All categories": "Vladimir Putin"})
    habib_time = get_data("https://raw.githubusercontent.com/FunnyUnicorn/NewPythonProject/main/KhabibTimeline4Years.csv").drop(labels = ['Week'],axis = 0).rename(columns={"Category: All categories": "Khabib Nurmagomedov"})
    gosling_time = get_data("https://raw.githubusercontent.com/FunnyUnicorn/NewPythonProject/main/GoslingTimeline4Years.csv").drop(labels = ['Week'],axis = 0).rename(columns={"Category: All categories": "Ryan Gosling"})
    morgen_time.loc[(morgen_time['Morgenstern'] == '<1'), 'Morgenstern'] = "0"
    morgen_time['Morgenstern'] = morgen_time['Morgenstern'].astype('int')
    putin_time.loc[(putin_time['Vladimir Putin'] == '<1'), 'Vladimir Putin'] = "0"
    putin_time['Vladimir Putin'] = putin_time['Vladimir Putin'].astype('int')
    habib_time.loc[(habib_time['Khabib Nurmagomedov'] == '<1'), 'Khabib Nurmagomedov'] = "0"
    habib_time['Khabib Nurmagomedov'] = habib_time['Khabib Nurmagomedov'].astype('int')
    gosling_time.loc[(gosling_time['Ryan Gosling'] == '<1'), 'Ryan Gosling'] = "0"
    gosling_time['Ryan Gosling'] = gosling_time['Ryan Gosling'].astype('int')

    all = pd.concat([morgen_time, gosling_time, putin_time, habib_time], axis=1)


    choice = st.multiselect("Choice", ["Morgenstern", "Vladimir Putin", 'Khabib Nurmagomedov', "Ryan Gosling"])

    fig1 = px.line(all, x=all.index, y=choice, title = 'Popularity dynamics')
    fig1.update_xaxes(rangeslider_visible=True, rangeselector=dict(
        buttons=list([
            dict(count=1, label="1 month", step="month", stepmode="backward"),
            dict(count=6, label="6 months", step="month", stepmode="backward"),
            dict(count=1, label="1 year", step="year", stepmode="backward"),
            dict(count=2, label="2 years", step="year", stepmode="backward"),
            dict(step="all")])))
    st.plotly_chart(fig1)

    st.header('Linear regression.')

    st.write('Here we will try to predict changes in popularity indices using the linear regression method.')
    pred = st.selectbox("Prediction choice", ["Morgenstern", "Vladimir Putin", 'Khabib Nurmagomedov', "Ryan Gosling"])
    y = np.array(all[pred])
    x = np.array(list(range(1, len(y)+1))).reshape((-1,1))
    model = LinearRegression().fit(x,y)
    blank = []
    for i in range(len(x+1)):
        blank.append(model.intercept_ + i * model.coef_[0])

    all['prediction'] = blank
    fig4 = px.line(all, x=all.index, y=[pred, 'prediction'], title='Prediction')
    fig4.update_xaxes(rangeslider_visible=True, rangeselector=dict(
        buttons=list([
            dict(count=1, label="1 month", step="month", stepmode="backward"),
            dict(count=6, label="6 months", step="month", stepmode="backward"),
            dict(count=1, label="1 year", step="year", stepmode="backward"),
            dict(count=2, label="2 years", step="year", stepmode="backward"),
            dict(step="all")])))
    st.plotly_chart(fig4)

    j = st.slider("Popularity in how many periods (weeks) do you want to see?", min_value=1, max_value=100)
    next = blank[-1] + j * model.coef_[0]
    st.write('Predicted popularity: ' + str(next))

    st.header('Statistical data.')

    st.write('Here we calculate the mean and variance measures for our popularity indices. Then we can look at how they have changed over time.')

    matrix = pd.DataFrame({'Morgenstern' : []})
    for element in ["Morgenstern", "Vladimir Putin", 'Khabib Nurmagomedov', "Ryan Gosling"]:
        avg = np.mean(np.array(all[element]))
        avg_square = np.mean(np.array(all[element]) ** 2)
        variance = avg_square - avg ** 2
        matrix[element] = [avg, avg_square, variance]
    matrix.index = ['E(x)', 'E(x^2)', 'Var(x)']
    matrix
    b = []
    choice2 = st.selectbox("Choose the celebrity", ["Morgenstern", "Vladimir Putin", 'Khabib Nurmagomedov', "Ryan Gosling"])
    avg_graph = []
    avg_square = []
    variance_graph = []
    for i in range(len(all[choice2])):
        znach = all[choice2][:i+1]
        avg_graph.append(np.mean(np.array(znach)))
        avg_square.append(np.mean(np.array(znach) ** 2))
        variance_graph.append(np.mean(np.array(znach) ** 2) - np.mean(np.array(znach))**2)
    historical_data = pd.DataFrame({'Average':avg_graph, 'Variance': variance_graph})
    fig3 = px.bar(historical_data, x = all.index, y = ['Average', 'Variance'], barmode="overlay")
    st.plotly_chart(fig3)

    st.write('Then we can calculate the correlation matrix for the popularity indices.')

    for element2 in ["Morgenstern", "Vladimir Putin", 'Khabib Nurmagomedov', "Ryan Gosling"]:
        for element3 in ["Morgenstern", "Vladimir Putin", 'Khabib Nurmagomedov', "Ryan Gosling"]:
            cov = np.sum((np.array(all[element2])-matrix[element2]['E(x)']) * (np.array(all[element3])-matrix[element3]['E(x)'])) / len(all[element2])
            corr = cov/((matrix[element2]['Var(x)'] ** 0.5)*(matrix[element3]['Var(x)'] ** 0.5))
            b.append(corr)

    corr_matrix = pd.DataFrame({'Morgenstern': []})
    corr_matrix["Morgenstern"] = b[0:4]
    corr_matrix["Vladimir Putin"] = b[4:8]
    corr_matrix['Khabib Nurmagomedov'] = b[8:12]
    corr_matrix["Ryan Gosling"] = b[12:16]
    corr_matrix.index = ["Morgenstern", "Vladimir Putin", 'Khabib Nurmagomedov', "Ryan Gosling"]
    corr_matrix


    st.header('World data.')
    st.write('Pick a famous person and look at the distribution of searches around the world. The map shows only the countries that have data. The brighter the light, the more popular the person is.')

    celebrity = st.selectbox("Choose a celebrity!", ['Vladimir Putin', 'Morgenstern', 'Ryan Gosling', 'Khabib Nurmagomedov'])

    world = gpd.read_file("https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json")
    world['name'] = np.where((world.name == 'United States of America'), "United States", world.name)
    if celebrity == 'Morgenstern':
        map_data = get_data("https://raw.githubusercontent.com/FunnyUnicorn/NewPythonProject/main/MorgensternMap.csv")
    elif celebrity == 'Vladimir Putin':
        map_data = get_data("https://raw.githubusercontent.com/FunnyUnicorn/NewPythonProject/main/VladimirPutinMap.csv")
    elif celebrity == 'Ryan Gosling':
        map_data = get_data("https://raw.githubusercontent.com/FunnyUnicorn/NewPythonProject/main/RyanGoslingMap.csv")
    elif celebrity == "Khabib Nurmagomedov":
        map_data = get_data("https://raw.githubusercontent.com/FunnyUnicorn/NewPythonProject/main/KhabibMap.csv")
    map = map_data.drop(['Country'])
    map['name'] = map.index
    map.loc[(map['Category: All categories'] == '<1'), 'Category: All categories'] = "0"
    map = map[map['Category: All categories'].notna()]
    map['Category: All categories'] = map['Category: All categories'].astype('int')
    res = world.merge(map, on=["name"])
    

    answer = st.selectbox("Would you like to see the popularity of a selected celebrity depending on the countries income? The calculation may take some time.", ['No', 'Yes'])
    if answer == 'Yes':
        def info_json(code):
            entrypoint = "http://api.worldbank.org/v2/country/" + code + "?format=json"
            params = {'format': 'json'}
            r = requests.get(entrypoint)
            a = r.json()[1]
            return a[0]["incomeLevel"]['value']

        income_list = []
        for element in res['id']:
            d = info_json(element)
            income_list.append(d)
        res['income'] = income_list

        income_popularity_corr = {'High income': np.array(np.mean(res[res['income'] == 'High income']["Category: All categories"])), 'Upper middle income':np.array(np.mean(res[res['income'] == 'Upper middle income']["Category: All categories"])), 'Lower middle income':np.array(np.mean(res[res['income'] == 'Lower middle income']["Category: All categories"]))}
        income_popularity_corr = pd.DataFrame(data = income_popularity_corr, index=['Popularity of ' + celebrity])
        income_popularity_corr

    connection = sqlite3.connect("population.sqlite")
    c = connection.cursor()

    def get_SQL(country):
        a = c.execute('''
        SELECT country, population, year FROM population_years
        WHERE country == (?) and year == 2010
        ''', [country]).fetchall()
        return a

    def sum_SQL():
        a = c.execute('''
        SELECT population FROM population_years
        WHERE year == 2010
        ''').fetchall()
        return a

    st.header('New index.')

    st.write('We will try to create a new popularity index that takes into account the number of Internet requests in relation to the percentage of the population in a given country, based on the world population. In other words, the formula will be:')
    st.latex(r'''\text{New index} = \dfrac{\text{Population of the country}}{\text{World population}} \cdot \text{Old index}''')
    blank2 = []
    t = sum_SQL()
    for element in t:
        if type(element[0]) == float:
            blank2.append(element[0])
    sum = np.sum(np.array(blank2))
    sort = res.sort_values(by = 'name')
    index_country = st.selectbox('Pick one country', sort['name'])
    ans = get_SQL(index_country)[0][1] * sort[sort['name'] == index_country]['Category: All categories']/ sum
    ans.index = [0]
    st.write("The new index for " + index_country + " is equal to " + str(ans[0]) + '.')
