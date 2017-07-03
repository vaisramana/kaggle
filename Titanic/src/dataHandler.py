


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
    train = pd.read_csv("../input/train.csv")
    train = simplifyFeatures(train)
    return train

