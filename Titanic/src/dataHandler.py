


import numpy as np
import pandas as pd

"""
Data Dictionary
Variable	Definition	Key
survival 	Survival 	0 = No, 1 = Yes
pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
sex 	Sex 	
Age 	Age in years 	
sibsp 	# of siblings / spouses aboard the Titanic 	
parch 	# of parents / children aboard the Titanic 	
ticket 	Ticket number 	
fare 	Passenger fare 	
cabin 	Cabin number 	
embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiances were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
"""


                
def simplifyAges(trainData):
    trainData.Age = trainData.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    groupNames = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(trainData.Age, bins, labels=groupNames)
    trainData.Age = categories
    return trainData

def simplifyCabins(trainData):
    trainData.Cabin = trainData.Cabin.fillna('N')
    trainData.Cabin = trainData.Cabin.apply(lambda x: x[0])
    return trainData

def simplifyFares(trainData):
    trainData.Fare = trainData.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(trainData.Fare, bins, labels=group_names)
    trainData.Fare = categories
    return trainData

def formatName(trainData):
    trainData['Lname'] = trainData.Name.apply(lambda x: x.split(' ')[0])
    trainData['NamePrefix'] = trainData.Name.apply(lambda x: x.split(' ')[1])
    return trainData    
    
def dropFeatures(trainData):
    return trainData.drop(['Ticket', 'Name', 'Embarked'], axis=1)




def simplifyFeatures(trainData):
    trainData = simplifyAges(trainData)
    trainData = simplifyCabins(trainData)
    trainData = simplifyFares(trainData)
    trainData = formatName(trainData)
    #trainData = dropFeatures(trainData)
    return trainData

def dataLoader():
    # reading train data
    train = pd.read_csv('../input/train.csv')
    
    # reading test data
    test = pd.read_csv('../input/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)
    

    # merging train data and test data for future feature engineering
    combinedData = train.append(test)
    combinedData.reset_index(inplace=True)
    combinedData.drop('index', inplace=True, axis=1)
    
    return combinedData


def status(feature):
    print("Processing %s ok" % (feature)) 


# Extracting the passenger titles
def getTitles(combinedData):
    # we extract the title from each name
    combinedData['Title'] = combinedData['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combinedData['Title'] = combinedData.Title.map(Title_Dictionary)
    status('title')
    return combinedData

# a function that fills the missing values of the Age variable
def fillAges(row, grouped_median):
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 1, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 1, 'Mrs']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['female', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['female', 1, 'Royalty']['Age']

    elif row['Sex']=='female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 2, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 2, 'Mrs']['Age']

    elif row['Sex']=='female' and row['Pclass'] == 3:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 3, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 3, 'Mrs']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 1, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 1, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['male', 1, 'Royalty']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 2, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 2, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 2, 'Officer']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 3, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 3, 'Mr']['Age']

            
# Processing the ages
def processAges(combinedData):
    grouped_train = combinedData.head(891).groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train.median()

    grouped_test = combinedData.iloc[891:].groupby(['Sex','Pclass','Title'])
    grouped_median_test = grouped_test.median()

    combinedData.head(891).Age = combinedData.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    combinedData.iloc[891:].Age = combinedData.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    status('ages')
    return combinedData


# This function drops the Name column since we won't be using it anymore because we created a Title column.
def processNames(combinedData):
    # we clean the Name variable
    combinedData.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combinedData['Title'],prefix='Title')
    combinedData = pd.concat([combinedData,titles_dummies],axis=1)
    
    # removing the title variable
    combinedData.drop('Title',axis=1,inplace=True)
    
    status('names')
    return combinedData



# This function simply replaces one missing Fare value by the mean.
def processFares(combinedData):
    # there's one missing fare value - replacing it with the mean. 
    #status('test-fare-1')
    #combinedData.info()
    
    tmp = combinedData.head(891).Fare.fillna(combinedData.head(891).Fare.mean())
    combinedData.head(891).Fare = tmp
    #status('test-fare-2')
    tmp = combinedData.loc[891:].Fare.fillna(combinedData.loc[891:].Fare.mean())
    combinedData.loc[891:].Fare = tmp

    #combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    #combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)
    
    status('fare')
    return combinedData

# This functions replaces the two missing values of Embarked with the most frequent Embarked value.
def processEmbarked(combinedData):
    # two missing embarked values - filling them with the most frequent one (S)
    combinedData.head(891).Embarked.fillna('S', inplace=True)
    combinedData.iloc[891:].Embarked.fillna('S', inplace=True)
    
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combinedData['Embarked'],prefix='Embarked')
    combinedData = pd.concat([combinedData,embarked_dummies],axis=1)
    combinedData.drop('Embarked',axis=1,inplace=True)
    
    status('embarked')
    return combinedData


# This function replaces NaN values with U (for Unknow). It then maps each Cabin value to the first letter. Then it encodes the cabin values using dummy encoding again.
def processCabin(combinedData):
    # replacing missing cabins with U (for Uknown)
    combinedData.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combinedData['Cabin'] = combinedData['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combinedData['Cabin'], prefix='Cabin')
    
    combinedData = pd.concat([combinedData,cabin_dummies], axis=1)
    
    combinedData.drop('Cabin', axis=1, inplace=True)
    
    status('cabin')
    return combinedData

# This function maps the string values male and female to 1 and 0 respectively.
def processSex(combinedData):
    # mapping string values to numerical one 
    combinedData['Sex'] = combinedData['Sex'].map({'male':1,'female':0})
    
    status('sex')
    return combinedData

# This function encodes the values of Pclass (1,2,3) using a dummy encoding.
def processPclass(combinedData):
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combinedData['Pclass'], prefix="Pclass")
    
    # adding dummy variables
    combinedData = pd.concat([combinedData,pclass_dummies],axis=1)
    
    # removing "Pclass"
    combinedData.drop('Pclass',axis=1,inplace=True)
    
    status('pclass')
    return combinedData

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'

# This functions preprocess the tikets first by extracting the ticket prefix. When it fails in extracting a prefix it returns XXX.
# Then it encodes prefixes using dummy encoding.
def processTicket(combinedData):
    # Extracting dummy variables from tickets:

    combinedData['Ticket'] = combinedData['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combinedData['Ticket'], prefix='Ticket')
    combinedData = pd.concat([combinedData, tickets_dummies], axis=1)
    combinedData.drop('Ticket', inplace=True, axis=1)

    status('ticket')
    return combinedData

"""
This function introduces 4 new features:

    FamilySize : the total number of relatives including the passenger (him/her)self.
    Sigleton : a boolean variable that describes families of size = 1
    SmallFamily : a boolean variable that describes families of 2 <= size <= 4
    LargeFamily : a boolean variable that describes families of 5 < size
"""

def processFamily(combinedData):
    # introducing a new feature : the size of families (including the passenger)
    combinedData['FamilySize'] = combinedData['Parch'] + combinedData['SibSp'] + 1
    
    # introducing other features based on the family size
    combinedData['Singleton'] = combinedData['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combinedData['SmallFamily'] = combinedData['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
    combinedData['LargeFamily'] = combinedData['FamilySize'].map(lambda s: 1 if 5<=s else 0)
    
    status('family')
    return combinedData
    
def dataHandler():
    combinedData = dataLoader()
    combinedData = getTitles(combinedData) 
    combinedData = processAges(combinedData)
    combinedData = processNames(combinedData)
    combinedData = processFares(combinedData)
    combinedData = processEmbarked(combinedData)
    combinedData = processCabin(combinedData)
    combinedData = processSex(combinedData)
    combinedData = processPclass(combinedData)
    combinedData = processTicket(combinedData)
    combinedData = processFamily(combinedData)
    return combinedData


    
    
