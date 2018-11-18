import numpy as np
from numpy import genfromtxt
import pandas as pd
import scipy.sparse as sp
import os

exported_data_file = "u.formated_data.csv"

uData = pd.read_csv(filepath_or_buffer="u.data.csv", delimiter=r"\s?\|\s?")
uItem = pd.read_csv(filepath_or_buffer="u.item.csv", delimiter=r"\s?\|\s?")
uUser = pd.read_csv(filepath_or_buffer="u.user.csv", delimiter=r"\s?\|\s?")
dataMatrix = sp.coo_matrix((uData["rating"],(uData["user.id"]-1,uData["item.id"]-1)))

def movielens_to_MLP_data():
    #########################################
    # ITEMS : size = 19
    items_categorie_df = uItem.iloc[:, 5:24]  # Garder seulement les catégorie des films
    items_categorie_matrix = items_categorie_df.as_matrix()  # Convertir en matrice

    #########################################
    # USERS : size = 23
    '''Convert gender df to nparray'''
    age_df = uUser.iloc[:, 1]
    normalized_age_df = (age_df - age_df.min()) / (age_df.max() - age_df.min())
    age_array = normalized_age_df.as_matrix()
    age_col = np.reshape(age_array, (len(age_array), 1))

    '''Convert gender df to nparray'''
    gender_df = uUser.iloc[:, 2]
    gender_array = (gender_df.replace("M", 0).replace("F", 1)).as_matrix()
    gender_col = np.reshape(gender_array, (len(gender_array), 1))

    '''Convert job df to nparray'''
    job_df = uUser.iloc[:, 3]
    # print(job_df.unique())
    job_one_hot_idx_mapping = {
        "administrator": 0,
        "artist": 1,
        "doctor": 2,
        "educator": 3,
        "engineer": 4,
        "entertainment": 5,
        "executive": 6,
        "healthcare": 7,
        "homemaker": 8,
        "lawyer": 9,
        "librarian": 10,
        "marketing": 11,
        "none": 12,
        "other": 13,
        "programmer": 14,
        "retired": 15,
        "salesman": 16,
        "scientist": 17,
        "student": 18,
        "technician": 19,
        "writer": 20
    }
    job_one_hot_idx_array = job_df.replace(job_one_hot_idx_mapping)
    n_row = len(job_df)
    n_col = len(job_one_hot_idx_mapping)
    job_matrix = np.zeros((n_row, n_col))
    job_matrix[np.arange(n_row), job_one_hot_idx_array] = 1

    '''Convert gender df to nparray'''
    user_description_matrix = np.concatenate((age_col, gender_col), axis=1)
    user_description_matrix = np.concatenate((user_description_matrix, job_matrix), axis=1)

    #########################################
    #     USER + ITEM + VOTE : 23 + 19 + 1 = 43 (size)
    formated_data = None
    for i, j, vote in zip(dataMatrix.row, dataMatrix.col, dataMatrix.data):
        user_info = user_description_matrix[i]
        item_info = items_categorie_matrix[j]
        user_item_info = np.append(user_info, item_info)
        user_item_vote_info = np.append(user_item_info, vote)
        user_item_vote_info = np.reshape(user_item_vote_info, (1, len(user_item_vote_info)))
        if formated_data is None:
            formated_data = user_item_vote_info
        else:
            formated_data = np.concatenate((formated_data, user_item_vote_info), axis=0)

    ########################################
    # Export data to csv
    n = formated_data.shape[1] - 1
    np.savetxt(exported_data_file, formated_data, fmt=' '.join(['%1.4f'] + ['%i'] * n))

    return formated_data

def is_already_exported() :
    file_path = "./" + exported_data_file
    return os.path.exists(file_path)

def get_X(formated_data) :
    return formated_data[:,:-1]

def get_Y(formated_data) :
    return formated_data[:,-1]


##############################################
#                 MAIN
##############################################

if is_already_exported() :
    formated_data = pd.read_csv(exported_data_file, delimiter=' ', header=None).values
else :
    formated_data = movielens_to_MLP_data()

print("X_data : " + str(get_X(formated_data)))
print("Y_data : " + str(get_Y(formated_data)))
