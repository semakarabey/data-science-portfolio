import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("17-googleplaystore.csv")
print(df.head())

print("-------------Columns' Names-----------")
print(df.columns)

print("------------Information of data frame----------")
print(df.info())

print("----------Showing null datas--------------")
print(df.isna().sum())

print("------------Statistics for data frame----------")
print(df.describe())

print("-------Check values for convertion----------")
print(df["Reviews"].str.isnumeric().sum())   # It means = could this string expression be converted  to numeric expression
print(df[~df["Reviews"].str.isnumeric()])    #  ~ uses to find the opposite


print("--------Cleaning that line----------")
df_clean = df.copy()
df_clean.drop(df_clean.index[10472], inplace=True)
print(df_clean["Reviews"].str.isnumeric().sum())



print("---------Converting from string to integer---------")
df_clean["Reviews"] = df_clean["Reviews"].astype(int)
print(df_clean.info())

print("----------Converting data types-------------")
print(df_clean["Size"].unique())   # There are two types named M and k so convert each value from M to k
df_clean["Size"] = df_clean["Size"].str.replace("M", "000")
df_clean["Size"] = df_clean["Size"].str.replace("k", "")
print(df_clean["Size"].unique())
df_clean["Size"] = df_clean["Size"].replace("Varies with device", np.nan)
df_clean["Size"]=df_clean["Size"].astype(float)
print(df_clean["Size"])

print("----------Fixing columns of Installs and Price---------")
print(df_clean["Installs"].unique())  # delete + sign
print(df_clean["Price"].unique())     # delete $ and , sign

chars_to_columns = ["+",",","$"]
cols_to_clean = ["Installs","Price"]

for item in chars_to_columns:
    for cols in cols_to_clean:
        df_clean[cols] = df_clean[cols].str.replace(item,"")


df_clean["Installs"] = df_clean["Installs"].astype(int)
df_clean["Price"] = df_clean["Price"].astype(float)

print(df_clean["Installs"].unique())  
print(df_clean["Price"].unique()) 
print(df_clean.info())


print("-----------Fixing the datas------------")
print(df_clean["Last Updated"].unique())
df_clean["Last Updated"] = pd.to_datetime(df_clean["Last Updated"])
print(df_clean.head())

print("--------------------------------------------------------------------------------------------")

df_clean["Day"] = df_clean["Last Updated"].dt.day
df_clean["Month"] = df_clean["Last Updated"].dt.month
df_clean["Year"] = df_clean["Last Updated"].dt.year

print(df_clean.head())

print("-----------Drop Duplicates------------")
print(df_clean[df_clean.duplicated("App")].shape)
df_clean.drop_duplicates(subset="App", keep="first", inplace=True)
print(df_clean.shape)


print("-------------the result is wrong just understand how we can do---------- ")
numeric_features = [feature for feature in df_clean.columns if df_clean[feature].dtype != 'O']
categorical_features = [feature for feature in df_clean.columns if df_clean[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


print("-----------Using Group By------------")
df_cat_installs = df_clean.groupby(['Category'])['Installs'].sum().sort_values(ascending=False).reset_index()
df_cat_installs["Installs"] = df_cat_installs["Installs"]/1000000000
print(df_cat_installs)

print("-------------Exercises----------------")
df2 = df_cat_installs.head()
plt.figure(figsize=(10,5))
sns.barplot(x="Installs", y="Category", data=df2)
plt.show()

# Top 5 most installed Apps in each popular Category
df_app_category = df_clean.groupby(['Category' ,'App'])['Installs'].sum().reset_index()
df_app_category = df_app_category.sort_values('Installs', ascending = False)
apps = ['GAME', 'COMMUNICATION', 'PRODUCTIVITY', 'SOCIAL' ,'TOOLS']
sns.set_context("poster")
sns.set_style("darkgrid")

plt.figure(figsize=(40,30))

for i,app in enumerate(apps):
    df3 = df_app_category[df_app_category.Category == app]

    df_top_5 = df3.head(5)
    plt.subplot(4,2,i+1)
    sns.barplot(data= df_top_5,x= 'Installs' ,y='App' )
    plt.xlabel('Installation in Millions')
    plt.ylabel('')
    plt.title(app,size = 20)
    
plt.tight_layout()
plt.show()


df_clean['Android Ver']=df_clean['Android Ver'].replace('and up', '', regex=True).replace('Varies with device', '', regex=True).replace('W', '', regex=True).replace('', np.nan)
print(df_clean['Android Ver'])
print(df_clean['Android Ver'].value_counts())