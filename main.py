from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/junaidqazi/DataSets_Practice_ScienceAcademy/master/City_of_Chicago_Payroll_Data.csv')


x = df[['Department', 'Job Titles','Salary or Hourly','Annual Salary']]

x = x[x['Salary or Hourly'] == 'Salary'].reset_index()

y = x[['Annual Salary']].applymap(lambda x: x.replace('$',""))

y = y['Annual Salary'].astype("float")

x = x[['Department', 'Job Titles']]

departamentos = set(x['Department'])
departamentos_ = {}
i = 1
for departamento in list(departamentos):
    departamentos_[i] = departamento
    i += 1

chaves = list(departamentos_.keys())
valores = list(departamentos_.values())

departamentos = pd.DataFrame({'id_departamento':chaves,'nm_departamento':valores})


job_titles = set(x['Job Titles'])
job_titles_ = {}
i = 1
for job_title in list(job_titles):
    job_titles_[i] = job_title
    i += 1

chaves = list(job_titles_.keys())
valores = list(job_titles_.values())

cargos = pd.DataFrame({'id_cargo':chaves,'nm_cargo':valores})

x = pd.merge(x,departamentos, left_on='Department', right_on='nm_departamento')
x = pd.merge(x, cargos,left_on='Job Titles', right_on='nm_cargo')
print(x.head())

x = x[['id_departamento','id_cargo']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

classifier = DecisionTreeRegressor()
classifier.fit(x_train, y_train)
preds_val = classifier.predict(x_test)

print(y_test)
print(preds_val)
# mae = mean_absolute_error(y_train, preds_val)

# print(mae)