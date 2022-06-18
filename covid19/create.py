import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
from datetime import timedelta
''' for notebook in colab
from google.colab import files
files.upload()
'''   
import warnings
import pickle
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

models =[] # will hold 4 model names
#Importing the raw data

print('upload raw data from files')
print('if a figure pops up, close it to move forward...')
try:
    encounters_df = pd.read_csv(
    "raw_data/encounters.csv",
    usecols=["PATIENT", "START", "REASONCODE", "CODE"],
    parse_dates=["START"],
    )
except FileNotFoundError:
    print("File not found: raw_data/encounters.csv")
    exit()
except Exception:
    print("Some other exception")
try:
    patients_df = pd.read_csv(
    "raw_data/patients.csv",
    usecols=["Id", "RACE", "GENDER", "BIRTHDATE", "DEATHDATE"],
    parse_dates=["BIRTHDATE", "DEATHDATE"],
    )
except FileNotFoundError:
    print("File not found: raw_data/patients.csv")
# Find all patients that were admitted to the hospital because of Covid-19

covid_encounters_df = encounters_df[
    encounters_df.REASONCODE.isin([840539006])
    | encounters_df.CODE.isin([1505002, 305351004])
    ]
#print(covid_encounters_df)
# merge
covid_patients_df = covid_encounters_df.merge(
    patients_df, left_on="PATIENT", right_on="Id"
    ).drop(["Id"], axis=1)
covid_patients_df["DIED"] = (~pd.isnull(covid_patients_df.DEATHDATE)).astype(int)
covid_patients_df["AGE"] = (
    (covid_patients_df.START.dt.tz_localize(None) - covid_patients_df.BIRTHDATE).dt.days
    / 365.2425
).round(1)

#Dataset for modeling, keep relevant columns only
covid_patients_df =covid_patients_df.loc[:,('GENDER','RACE','AGE','DIED')]

sns.set_theme(style="whitegrid")
sns.relplot(
    data=covid_patients_df,
    x="AGE", y="RACE", col="DIED",
    )
plt.show()
#Here checked, no missing values, otherwise use sciikit imputater
#Use seaborn for initial visualizations
# Apply the default theme

dis = sns.displot(covid_patients_df, x="AGE", hue="DIED", multiple="dodge")
#dis.set(title = "Comparision of Age Histograms DIED and SURVIVED")
survived_mean_age = covid_patients_df.query('DIED == 0')['AGE'].mean().round()
died_mean_age = covid_patients_df.query('DIED == 1')['AGE'].mean().round()
plt.axvline(x=survived_mean_age, color='green', label=" SURVIVED mean: "+str(survived_mean_age)) 
plt.axvline(x=died_mean_age, color='red', label="DIED mean:"+str(died_mean_age))
plt.legend(bbox_to_anchor=(0.1, .95, 1, 0), loc=2, ncol=2, mode="expand", borderaxespad=0)
plt.savefig("result_data/age_histogram.png")
plt.show()

cat = sns.catplot(data=covid_patients_df, kind="violin", x="RACE", y="AGE", hue="DIED", split=True)
cat.set(title = "Age distribution comparisons side by side")
plt.savefig("result_data/age_violin_plot.png")
plt.show()

race_df = covid_patients_df.groupby("RACE").mean().reset_index()
sns.set_theme(style="whitegrid")
sns.barplot(x=race_df.RACE, y=race_df.DIED)
plt.savefig("result_data/age_bar_plot.png")
plt.show()

sns.catplot(data=covid_patients_df, kind="violin", x="GENDER", y="AGE", hue="DIED", split=True)
plt.savefig("result_data/gender_violin_plot.png")
plt.show()
########################## Modeling ####################
## Independent variables and dependent one
X = covid_patients_df.iloc[:, :-1]
y = covid_patients_df.iloc[:, -1]


# Transform all categorical variables into numeric, usung One Hot Encoder
temp_df = covid_patients_df[['GENDER', 'RACE', 'AGE', 'DIED']]
temp_df.loc[:,'GENDER'] = (temp_df.GENDER == 'M').astype(int)
X = temp_df.iloc[:, :-1]
y = temp_df.iloc[:, -1]

#print(X)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['RACE'])], remainder='passthrough')
cX = ct.fit_transform(X)

filename = 'result_data/column_transformer_model.sav' # for prediction, save this transformer
pickle.dump(ct, open(filename, 'wb'))
#print(cX)
## back to dataframe:
dX =pd.DataFrame(cX,columns=ct.get_feature_names_out())
print(ct.get_feature_names_out())
print(dX)

## Transform the dependent variable (DIED: 1, SURVIDED: 0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dX, y, test_size = 0.25, random_state = 0)

## Feature Scaling (skipped this) optional as the numbers look good, not too big, tried but didn't do any better
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#X_train
from sklearn.metrics import confusion_matrix, accuracy_score
## Model 1: Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)

# save the model to disk
filename = 'result_data/log_reg_model.sav'
models.append(filename)
pickle.dump(model, open(filename, 'wb'))
# view the coefficiencies
print(model.coef_, model.intercept_)

## Model 2: Training the Decision Tree model on the Training set
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
from sklearn import tree
model.fit(X_train, y_train)
#print(tree.export_text(model, feature_names=list(X_train.columns))) # <-- view the tree
# save the model to disk
filename = 'result_data/decision_tree_model.sav'
models.append(filename)
filename = 'decision_tree_model.sav'
pickle.dump(model, open(filename, 'wb'))

## Model 3: Training the K-Nearest Neighbor model on the Training set

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(X_train, y_train)
# save the model to disk
filename = 'result_data/knn_model.sav'
models.append(filename)
pickle.dump(model, open(filename, 'wb'))


''' #try different n_neighbors
for i in range(3, 100, 5):
    model = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
    k_model = model.fit(X_train, y_train)
    y_pred = k_model.predict(X_test)
    print(i, accuracy_score(y_test, y_pred))
'''

## Model 4: Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
model = SVC(kernel = 'rbf', random_state = 0)
model.fit(X_train, y_train)
# save the model to disk
filename = 'result_data/svm_model.sav'
models.append(filename)
pickle.dump(model, open(filename, 'wb'))


## Making the Confusion Matrix (Test the last model that was run from the above 4 models)

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_scores = []
for model in models:
    loaded_model = pickle.load(open(model, 'rb'))
    y_pred = loaded_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy_scores.append(round(accuracy_score(y_test, y_pred)*100,2))
    print(cm)
for i in range(len(accuracy_scores )):
    print(str( accuracy_scores[i]) +": " + models[i])
print("score average from 4 ML models: " + str(round(sum(accuracy_scores)/len(accuracy_scores ),2)))
print(" 4 trained models are saved in result_data/ folder")

print("\n\ntry deep learning model ...")
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
model =keras.Sequential([
    keras.layers.Flatten(input_shape=(6,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
]) # Dense layers, 2, 4, 8, no sigmificant improvement on accurracy keras.layers.Dense(16, activation=tf.nn.relu),
#increase number nodes, no sigmificant improvement, keras.layers.Dense(32, activation=tf.nn.relu),
print("\n single CPU, the fit would likely take a few minutes ....")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0, use_multiprocessing=False) 
test_loss, test_acc = model.evaluate(X_test, y_test)
print ("DL model, test_loss: "+str(test_loss)) #test_loss: 0.5103273391723633
print ("DL model, test_acc: "+str(test_acc))# #test_acc: 0.7635829448699951








