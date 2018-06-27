import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.isnull().sum()

def main(train):

    del train['PassengerId']
    del train['Embarked']
    del train['Ticket']
    
    train.Sex = train.Sex.map({'male':1, 'female':2})
    
    import re 

    def search_pattern(index):
        return re.search(',.*[/.\b]', train.Name[index])[0][2:-1]

    train['Social_name'] = [search_pattern(counter) for counter in range(train.shape[0])]
    
    train.Social_name.unique()

    train.Social_name.replace({"""Mrs. Martin (Elizabeth L""":'Mrs', 'Mme':'Miss', 'Ms':'Miss', 'the Countess':'Countess', 'Mr. Carl':'Mr', 'Mlle':'Miss'}, inplace = True)

    for x in range(len(train.Social_name.unique())):
        a = train.Social_name.unique()[x]
        b = x
        train.Social_name.replace({a:b}, inplace = True)

    del train['Name']

    grouped = train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median()

    pclass = grouped.index.labels[0]; sex = grouped.index.labels[1]; social_name = grouped.index.labels[2];


    for counter in range(len(grouped.index.labels[1])):
        train.loc[((train.Pclass == train.Pclass.unique()[pclass[counter]]) &
                  (train.Sex == train.Sex.unique()[sex[counter]]) &
                  (train.Social_name == train.Social_name.unique()[social_name[counter]])),
                  'Age'] = \
        train.loc[((train.Pclass == train.Pclass.unique()[pclass[counter]]) &
                  (train.Sex == train.Sex.unique()[sex[counter]]) &
                  (train.Social_name == train.Social_name.unique()[social_name[counter]])),
                  'Age'].fillna(value = grouped.values[counter])

    for x in range(len(train)):
        if pd.isnull(train.loc[x,'Cabin']):
            continue 
        else: 
            train.loc[x,'Cabin'] = train.loc[x,'Cabin'][0]

    train.Cabin.fillna('N',inplace = True)

    train = pd.concat([train, pd.get_dummies(train.Cabin)], axis = 1)

    del train['N']
    del train['Cabin']
    
    train.Age = train.Age.values.round().astype(int)
    train.Fare = train.Fare.round().astype(int)
    
    return train
    
    
df_train = pd.read_csv('train.csv')
df_train = main(df_train)
del df_train['T']

model_training = df_train.loc[:, df_train.columns != 'Survived']
model_testing = df_train.loc[:, 'Survived']

df_test = pd.read_csv('test.csv')
df_test.Fare.fillna(df_test.Fare.median(skipna = True), inplace = True)
df_test = main(df_test)

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

the_model = logistic.fit(model_training, model_testing)
results = the_model.predict(df_test)

final = pd.read_csv('test.csv', usecols = ['PassengerId'])
final['Survived'] = results

final = final.set_index('PassengerId')

final.to_csv('final.csv')
