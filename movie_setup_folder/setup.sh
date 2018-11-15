#download the 1 million rating database which we could want to train on
wget -nc http://files.grouplens.org/datasets/movielens/ml-1m.zip
#setting up the datafolder
unzip ml-1m.zip -d movie_dataset_1M
mv movie_dataset_1M/ml-1m/* movie_dataset_1M
rmdir movie_dataset_1M/ml-1m
#move the folder where you want to
mkdir /home/glytz/Datasets/movie_dataset_1M
mv movie_dataset_1M/* /home/glytz/Datasets/movie_dataset_1M

rmdir movie_dataset_1M

#download the 100k dataset
#download the 20 million rating database which we could want to train on
wget -nc http://files.grouplens.org/datasets/movielens/ml-100k.zip
#setting up the datafolder
unzip ml-100k.zip -d movie_dataset_100K
mv movie_dataset_100k/ml-100k/* movie_dataset_100k
rmdir movie_dataset_1M/ml-100k
#move the folder where you want to
mkdir /home/glytz/Datasets/movie_dataset_100K
mv movie_dataset_100K/* /home/glytz/Datasets/movie_dataset_100K
rmdir movie_dataset_100K
