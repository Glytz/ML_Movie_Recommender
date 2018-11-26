import numpy as np


def calculate_quadratic_error(predicted_votes, real_votes):
    # calculate mask to only calculate the quadratic error where we have common vote, to not include NA (0) values
    predicted_votes[predicted_votes == 0] = np.nan
    real_votes = real_votes.astype(float)
    real_votes[real_votes == 0] = np.nan
    quadratic_error = np.power((predicted_votes - real_votes), 2)
    return np.sqrt(np.nanmean(quadratic_error))


def calculate_abs_error(predicted_votes, real_votes):
    # calculate mask to only calculate the quadratic error where we have common vote, to not include NA (0) values
    predicted_votes[predicted_votes == 0] = np.nan
    real_votes = real_votes.astype(float)
    real_votes[real_votes == 0] = np.nan
    abs_error = np.abs(predicted_votes - real_votes)
    return np.nanmean(abs_error)

def create_mask(matrice):
    np.seterr(divide='ignore', invalid='ignore')
    return np.where(matrice == 0, 0, matrice / matrice)