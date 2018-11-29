# log6308_projet
#Dépendances

Python 3.6
Librairie = {Keras, Tensorflow, numpy, pandas, scipy, matplotlib, os, argparse}
# Je recommende d'utiliser Ubuntu comme OS, pour la plus grande simpliciter d'installation des composantes
#commande utile pour installer certaine librairie
numpy : pip3 install numpy
pandas : pip3 install pandas
scipy : pip3 install scipy
matplotlib : pip3 install scipy
tensorflow : pip3 install tensorflow-gpu
#pour faire fonctionner avec le gpu, veuillez suivre les instructions sur : 
# https://www.tensorflow.org/install/gpu

pip3 install keras


#Pour rouler le projet

Assurez-vous de ne pas modifier l'arborescence du projet, et que les fichiers u.data.csv, u.item.csv et u.user.csv sont dans le même dossier que les autres fichiers python 

#Commande pour rouler le projet

Pour rouler le projet, il faut rouler le fichier python3 main.py ex : python3 main.py

Vous pouvez utiliser plusieurs paramètres avec celui-ci soit : 
#Pour les boolean, 0==False et 1==True
--load_model : boolean qui décide si on charge ou non un modèles existant 
--train_model : boolean qui décide si on entraine le modèle
--use_data_v1 : boolean qui décide si on utilise la premières versions des données ou la deuxièmes
--model_path : str représentant le Chemin relatif où trouver le model si on à décider de le charger (le modèles en cours est saugarder dans models/classifier.h5)
--is_softmax boolean sur si le modèles est utilise softmax en sortie sinon un autre type
--seed : int représentant la 'seed' pour les fonctions aléatoire de numpy
--epochs : nombre d'épochs d'entrainement du modèle avant d'arrèter l'entrainement
--batch_size : int représentant la tailles de bath pour l'entrainement du modèle ainsi que les tests

#commande pour rouler le code sur ubuntu
#pour rouler le model1 :
python3 -W ignore main.py --load_model 1 --train_model 0 --use_data_v1 1 --model_path "/model1/classifier.h5" --is_softmax 0 --seed 1693214 --epochs 500 --batch_size 128
#pour rouler le model2 :
python3 -W ignore main.py --load_model 1 --train_model 0 --use_data_v1 1 --model_path "/model2/classifier.h5" --is_softmax 1 --seed 1693214 --epochs 500 --batch_size 128
#Pour rouler le model3 :
python3 -W ignore main.py --load_model 1 --train_model 0 --use_data_v1 0 --model_path "/model3/classifier.h5" --is_softmax 1 --seed 1693214 --epochs 500 --batch_size 128

#Pour entrainer le model avec le data1 :
python3 -W ignore main.py --load_model 0 --train_model 1 --use_data_v1 1 --is_softmax 1 --seed 1693214 --epochs 500 --batch_size 128

#Pour entrainer le model avec le data2 :
python3 -W ignore main.py --load_model 0 --train_model 1 --use_data_v1 0 --is_softmax 1 --seed 1693214 --epochs 500 --batch_size 128

#Pour continue l'entrainement sur le model 2 : 
python3 -W ignore main.py --load_model 1 --train_model 1 --use_data_v1 1 --model_path "/model2/classifier.h5" --is_softmax 1 --seed 1693214 --epochs 500 --batch_size 128




