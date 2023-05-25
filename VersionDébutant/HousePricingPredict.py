#! /usr/local/bin/python3
#-*- coding : utf-8-*

# =============================Note au nouvel arrivant : ==============================
# Bienvenue dans le premier programme de machine learnig créé entièrement par Louis Brochard
# Ce programme est destiné à calculer le prix de maisons situées dans la ville de Ames, dans l'Iowa. 
# C'est un premier programme pour Louis Brochard, c'est pourquoi les techniques sont relativement basiques 
# 	et c'est pourquoi vous trouverez beaucoup de commentaires tout au long du script. 
# Ce programme est destiné à servir de modèle pour en créer d'autres. Il a été conçu après consultation du MOOC Machine Learning de Kaggle. 
# La data est aussi issue du site Kaggle. 

# Les modules/packages dont nous auront besoin dans ce script : 
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor



# Pour un algorithme de ML il faut sectionner la data en deux parties : 
#		- une partie "trainning"
#		- et une partie "test" qui va servir à entrer des nouvel 
# 
#
# La partie "Trainning" va servir à entrainer l'algorithme, alors que la partie "Test" servira à le tester avec des nouvelles valeurs,
# L'algorithme va sortir des résultats dont l'on ne dispose pas pour la partie "Test". 

# De plus, la partie "Train" doit obligatoirement se split en deux : 
# 		- une première partie va servir à fit l'algorithme. 
#		- une deuxième partie "validation" va servir à tester la fiabilité de l'algorithme. 
#
# En résumé : 
# 		1ere partie = Train
#			- une première partie de Train pour fit l'algo, (on a les inputs et les outputs).
#			- une sedonde  partie pour tester sa fiabilité, (on a les inputs et les outputs). 
#		2eme partie = Test 
#			- on ne dispose que des inputs (qui sont de nouvelles valeurs, et on va tenter de trouver les outputs, que l'on n'a pas)



# Pour l'instant il s'agit de mettre en place un algorithme et de tester sa fiabilité. 
# Il faut alors la charger. 
train = pd.read_csv('train.csv')

# La data est très lourde et comporte de nombreux patterns. 
# Pour cet exemple nous n'allons en prendre qu'une partie et donc sélectionner des colonnes. 
# NB : il est possible d'afficher les intitulés de colonnes (et donc le nom de chaque feature) en entrant : train.columns

# Les features vont composer X 
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train[features]


# y va être composé de la colonne price (pour train, puisque test ne comporte pas de colonne SalePrice, c'est ce que l'on recherche). 
y = train.SalePrice

# La data est, pour l'instant, split en deux. Une partie X et une partie Y
# Il faut ensuite la split encore en deux (chaque composant sera split en deux) : une partie "train" et une partie "validation"

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)





#======================First Algorithm : DecisionTreeRegressor======================

print("On commence par opérer les prédictions à l'aide d'un 'DecisionTreeRegressor'... ")
iowa_model_v1 = DecisionTreeRegressor(random_state = 1)

iowa_model_v1.fit(train_X, train_y)

predictions_v1 = iowa_model_v1.predict(val_X)

# Après avoir fit un algorithme en fonction des valeurs de train_X et train_y, on peut prédire des prix de ventes pour val_X. 
# Le test de fiabilité de l'algorithme va consister à vérifier que les valeurs des prédictions sont les plus proches possibles de val_y. 
# On va opérer le test de mean_absolute_error 

val_mae = mean_absolute_error(val_y , predictions_v1)

print ("La MAE du DecisionTreeRegressor est de : ", val_mae)

# Si la MAE nous convient, on fait marcher l'algorithme sur la data de test.csv, pour obtenir des prédictions (avec un marge d'erreur représentée par la MAE). 


#====================== Second Algorithm : DecisionTreeRegressor w/ max_leaf_nodes=========================

print("On poursuit avec une prédiction des prix à l'aide du DecisionTreeRegressor et l'utilisation de l'outil max_leaf_nodes")

# L'utilité de l'ajout d'un nouvel outil est d'éviter l'overfitting ou l'underfitting
# Ces plaies correspondent à un arbre de décision trop (ou pas assez) profond, qui ont pour conséquence un algorithm qui fonctionne mal. 
# L'outil max_leaf_nodes va servir à tester la valeur de MAE en fonction de la profondeur de l'arbre de décision. 
# Ainsi on choisira un nombre de level qui apporte la plus faible MAE. 

# Il faut pour cela définir une fonction qui va donner la MAE en fonction du nombre de level : 

def get_mae (max_leaf_nodes, train_X, val_X, train_y, val_y) : 
	"""Fonction qui va donner la MAE en fonction d'une certaine profondeur de l'arbre de décision"""

	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes , random_state = 0)
	model.fit(train_X, train_y)
	predictions_v2 = model.predict(val_X)
	mae = mean_absolute_error(val_y, predictions_v2)
	return(mae) 


for max_leaf_nodes in [5, 50, 500, 5000] : 
	my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
	print("Pour un Max Leaf Nodes de : ", max_leaf_nodes, " la MAE est égale à : ", my_mae)

# On affiche ainsi toutes les MAE et on va pouvoir sélectionner le niveau de profondeur qui offre la plus faible MAE. 

best_tree_size = input("Quel niveau de profondeur (MaxLeafNodes) souhaitez vous appliquer ici ? : ")
best_tree_size = int(best_tree_size)

iowa_model_v2 = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 1)
iowa_model_v2.fit(X , y )
# On peut prendre toute la data sans la split, on aura pas besoin de validation puisqu'on sait déjà que notre arbre possède une profondeur parfaite. 

# Si ce modèle nous convient on peut directement prédire les prix qui ressortiront de test.csv
# On ne va pas le faire pour le moment 

val_mae_2 = get_mae(best_tree_size, train_X, val_X, train_y, val_y)
print("La MAE du DecisionTreeRegressor est de : ", val_mae)

#================================Third Algorithm : Random Forest================================

print("On va enfin finir par un algorithm de RandomForestRegressor...")

# L'utilité de cet algorithme est qu'il est censé trouver tout seul le bon équilibre entre underfitting et overfitting. 
# Nous allons le tester et comparer sa MAE aux autres.

iowa_model_v3 = RandomForestRegressor(random_state=1)
iowa_model_v3.fit(train_X, train_y)
predictions_v3 = iowa_model_v3.predict(val_X)
val_mae_3 = mean_absolute_error(val_y, predictions_v3)

print("La MAE du RandomForestRegressor est de : ", val_mae_3)


# On va donc afficher les MAE des trois techniques puis choisir laquelle on désire pour notre algorithme (logiquement celle qui a la plus petite MAE.)

print("Les différentes MAE sont : ")
print("DecisionTreeRegressor : ", val_mae)
print("DecisionTreeRegressor avec un contrôle de la profondeur : ", val_mae_2)
print("RandomForestRegressor : ", val_mae_3)


choixmethode = input('Tapez {1}, {2} ou {3} en fonction de la méthode chosie : ')
choixmethode = int(choixmethode)

test = pd.read_csv('test.csv')
test_X = test[features]

if choixmethode == 1 : 
	test_y = iowa_model_v1.predict(test_X)
	output = pd.DataFrame({'Id' : test.Id , 'SalePrice' : test_y})
	output.to_csv('my_submission.csv', index = False)
	print("Your submission of DecisionTreeRegressor was succesfully made.")
elif choixmethode == 2 :
	test_y = iowa_model_v2.predict(test_X)
	output = pd.DataFrame({'Id' : test.Id , 'SalePrice' : test_y})
	output.to_csv('my_submission.csv', index = False)
	print("Your submission of DecisionTreeRegressor with MaxLeafNodes was succesfully made.")
elif choixmethode == 3 : 
	test_y = iowa_model_v3.predict(test_X)
	output = pd.DataFrame({'Id' : test.Id , 'SalePrice' : test_y})
	output.to_csv('my_submission.csv', index = False)
	print("Your submission of RandomForestRegressor was succesfully made.")




