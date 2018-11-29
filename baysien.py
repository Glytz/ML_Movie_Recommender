import pandas as pd
import numpy as np
import utils
import random


def get_job_idx(job_title, df):
    return np.sort(df['job'].unique()).tolist().index(job_title)


def get_gender_idx(gender, df):
    return np.sort(df['gender'].unique()).tolist().index(gender)


def get_age_group_idx(age_group, df):
    return np.sort(df['age_group'].unique()).tolist().index(age_group)


def calculate_probabilities(dataframe, values_name="rating", column_name=None, column_key=None, original_df=None):
    df_votes = dataframe
    if column_name != None:
        df_votes = dataframe.loc[dataframe[column_name] == column_key]
        if df_votes.empty:
            return np.full((original_df["item.id"].max(), original_df["rating"].max()), 0.2)
    df_votes = df_votes.groupby(["item.id", values_name]).size()
    matrix_votes = df_votes.unstack(values_name)
    matrix_votes = matrix_votes.reindex(range(1, original_df.groupby(["item.id"]).size().shape[0] + 1))
    matrix_votes.fillna(0, inplace=True)
    nb_vote_par_film = matrix_votes.sum(axis=1)
    nb_vote_par_film = nb_vote_par_film + matrix_votes.shape[1]
    matrix_votes = matrix_votes + 1
    df_votes_par_film = matrix_votes.div(nb_vote_par_film, axis=0)
    return np.array(df_votes_par_film)


# Postuler l'indépendance des facteurs: Pr(V=v|A=a,G=g,J=j) = Pr(V=v)Pr(A=a|V=v)Pr(G=g|V=v)Pr(J=j|V=v)
# Selon Bayes : P(A|B) = P(B|A)P(A)/P(B). On peux transformer ainsi:
# Dans notre cas: Pr(V=v|A=a,G=g,J=j) = P(V) * P(V|J)P(V)/P(J) * P(V|G)P(V)/P(G) * P(V|A)P(V)/P(A)
def similarities(dataframe, job, gender, age, original_df):
    df = dataframe
    p_v = calculate_probabilities(df, "rating", original_df=original_df)

    p_v_j = calculate_probabilities(df, "rating", "job", job, original_df=original_df)
    p_j = calculate_probabilities(df, "job", original_df=original_df)
    job_idx = get_job_idx(job, original_df)
    p_j = p_j[:, job_idx]

    p_v_g = calculate_probabilities(df, "rating", "gender", gender, original_df=original_df)
    p_g = calculate_probabilities(df, "gender", original_df=original_df)
    gender_idx = get_gender_idx(gender, original_df)
    p_g = p_g[:, gender_idx]

    age_group = age // 10 + 1
    p_v_a = calculate_probabilities(df, "rating", "age_group", age_group, original_df=original_df)
    p_a = calculate_probabilities(df, "age_group", original_df=original_df)
    age_group_idx = get_age_group_idx(age_group, original_df)
    p_a = p_a[:, age_group_idx]

    # Bayes: P(A|B) = P(B|A)P(A)/P(B)
    p_j_v = p_v_j * p_j.reshape(len(p_j), 1)
    p_j_v = p_j_v / p_v

    p_g_v = p_v_g * p_g.reshape(len(p_g), 1)
    p_g_v = p_g_v / p_v

    p_a_v = p_v_a * p_a.reshape(len(p_a), 1)
    # if (p_a_v.shape[1] == 0):

    p_a_v = p_a_v / p_v

    # Pr(V=v|A=a,G=g,J=j) = Pr(V=v)Pr(A=a|V=v)Pr(G=g|V=v)Pr(J=j|V=v)
    P_V_A_G_J = p_v * p_a_v * p_g_v * p_j_v

    # normalize the rating of each row so it sum to 1
    votes_probabilities = P_V_A_G_J / P_V_A_G_J.sum(axis=1, keepdims=True)
    # Multiplication des valeur avec les index (ratings)
    ratings = (votes_probabilities * np.array([1, 2, 3, 4, 5])).sum(axis=1)
    return ratings


def printHighestValue(array, uItem, number_to_print):
    # we recommend the number_to_print highest values
    n_voisin = 10
    near_idx = np.argsort(array)[-number_to_print:]
    # near_idx = np.argsort(new_user_predicted_votes)[new_user_predicted_votes.size - n_voisin:]
    near_values = [array[i] for i in near_idx]
    # Imprimer les distance
    print("Rank \t Vote \t\t Movie Id \t Title")
    for i in range(0, len(near_idx)):
        idx = near_idx[i]
        id = idx + 1
        title = uItem.loc[uItem["movie id"] == id]["movie title"].iloc[0]
        vote = near_values[i]
        print(str(len(near_idx) - i) + "\t" + str(vote) + "\t\t" + str(id) + "\t" + title)
    print("\n")


def cross_validation():
    uData = pd.read_csv(filepath_or_buffer="u.data.csv", delimiter=r"\s?\|\s?")
    #uItem = pd.read_csv(filepath_or_buffer="u.item.csv", delimiter=r"\s?\|\s?")
    uUser = pd.read_csv(filepath_or_buffer="u.user.csv", delimiter=r"\s?\|\s?")
    df = pd.merge(uData, uUser, left_on='user.id', right_on='id')
    df['age_group'] = df["age"] // 10 + 1

    number_of_users = df["user.id"].max()
    tot = 0
    absolute_error = 0
    quadratic_error = 0
    for i in range(1, number_of_users):
        user_to_delete = i
        train = df[df["user.id"] != user_to_delete]
        test = df.loc[df["user.id"] == user_to_delete]
        # get sex, occupation and age_group
        sex = (uUser["gender"])[user_to_delete - 1]
        occupation = uUser["job"][user_to_delete - 1]
        age = uUser["age"][user_to_delete - 1]
        rating = similarities(train, occupation, sex, age, df)
        y = 0
        for row in test.itertuples():
            y += 1
            item_id = row[2]
            vote = row[3]
            prediction = rating[item_id - 1]
            absolute_error += np.abs(prediction - vote)
            quadratic_error += np.power((prediction - vote), 2)
        tot += y
        y = 0
    quadratic_error /= tot
    quadratic_error = np.sqrt(quadratic_error)
    absolute_error /= tot
    return absolute_error, quadratic_error



