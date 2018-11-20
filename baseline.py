import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

def CalculateQuadraticError(predicted_votes, real_votes):
    # calculate mask to only calculate the quadratic error where we have common vote, to not include NA (0) values
    predicted_votes[predicted_votes == 0] = np.nan
    real_votes = real_votes.astype(float)
    real_votes[real_votes == 0] = np.nan
    quadratic_error = np.power((predicted_votes - real_votes), 2)
    return np.sqrt(np.nanmean(quadratic_error))


def CalculateAbsoluteError(predicted_votes, real_votes):
    # calculate mask to only calculate the quadratic error where we have common vote, to not include NA (0) values
    predicted_votes[predicted_votes == 0] = np.nan
    real_votes = real_votes.astype(float)
    real_votes[real_votes == 0] = np.nan
    abs_error = np.abs(predicted_votes - real_votes)
    return np.nanmean(abs_error)

def create_mask(matrice):
    np.seterr(divide='ignore', invalid='ignore')
    return np.where(matrice == 0, 0, matrice / matrice)

def baseline_cross_validation(original_graph, number_of_iteration):
    temp_abs_error = 0
    temp_quadratic_error = 0
    for i in range(number_of_iteration):
        index_to_remove = (np.random.randint(1, 100000, original_graph.shape) * create_mask(
            original_graph)).argsort(
            axis=1)[:, original_graph.shape[1] - 1]
        leave_one_out_graph = original_graph.copy()

        for j in range(len(index_to_remove)):
            leave_one_out_graph[j, index_to_remove[j]] = 0

        baseline_prediction = np.where(True,
                                       leave_one_out_graph.mean(axis=0),
                                       leave_one_out_graph)
        leave_one_out_predictions = baseline_prediction
        predicted_values = np.zeros(original_graph.shape[0])
        real_values = np.zeros(original_graph.shape[0])

        for j in range(len(index_to_remove)):
            predicted_values[j] = leave_one_out_predictions[j, index_to_remove[j]]
            real_values[j] = original_graph[j, index_to_remove[j]]

        temp_quadratic_error += CalculateQuadraticError(predicted_values, real_values)
        temp_abs_error += CalculateAbsoluteError(predicted_values, real_values)
    temp_abs_error /= number_of_iteration
    temp_quadratic_error /= number_of_iteration
    return temp_abs_error, temp_quadratic_error

def baseline_error():
    np.seterr(divide='ignore')
    uData = pd.read_csv(filepath_or_buffer="u.data.csv", delimiter=r"\s?\|\s?")
    uItem = pd.read_csv(filepath_or_buffer="u.item.csv", delimiter=r"\s?\|\s?")
    uUser = pd.read_csv(filepath_or_buffer="u.user.csv", delimiter=r"\s?\|\s?")
    dataMatrix = sp.coo_matrix((uData["rating"], (uData["user.id"] - 1, uData["item.id"] - 1)))
    dense_data_matrix = np.asarray(dataMatrix.todense())
    dense_data_matrix = dense_data_matrix.astype(float)
    # matrix with 0
    original_dense_matrix = dense_data_matrix.copy()
    dense_data_matrix[dense_data_matrix == 0] = np.nan
    # si 0, on met la moyenne de la colonne
    dense_data_matrix = np.where(np.isnan(dense_data_matrix),
                                 np.ma.array(dense_data_matrix, mask=np.isnan(dense_data_matrix)).mean(axis=0),
                                 dense_data_matrix)
    # ---------------------------------------------------------------------------------------------------------
    # Q1 (2 pts) Déterminez un point de comparaison pour la prévision de votes (une performance minimale)
    # ---------------------------------------------------------------------------------------------------------

    baseline_abs_error, baseline_quadratic_error = baseline_cross_validation(original_graph=original_dense_matrix,
                                                                                 number_of_iteration=5)
    return baseline_abs_error, baseline_quadratic_error
