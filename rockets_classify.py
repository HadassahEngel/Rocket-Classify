import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

C1 = 1
C2 = 4
C3 = 7
C4 = 10


def create_data():
    df = pd.read_csv('train.csv')
    df = df[(df["class"] == C1) | (df["class"] == C2) | (df["class"] == C3) | (df["class"] == C4)]
    train_df = df.sample(frac=0.8)
    merge_df = df.merge(train_df, how='left', indicator=True)
    check_df = merge_df[merge_df["_merge"] == 'left_only'].iloc[:, :-1]
    return train_df, check_df


def classify(row):
    x = row[2::7]
    z = row[4::7]
    xv = row[5::7]
    zv = row[7::7]
    if max(z) > 7500:
        return C2
    x = row[2::7]
    z = row[5::7]
    x = x.to_numpy()
    mx = np.argmax(x)
    if z[mx] > 350:
        return C2
    return C1


def cal_score(d):
    precision = d[(C2, C2)] / (d[(C2, C2)] + d[(C2, C1)])
    recall = d[(C1, C1)] / (d[(C1, C1)] + d[(C1, C2)])
    return 2 * ((precision * recall) / (precision + recall))


def get_kinetic_energy(train_set):
    n = train_set
    for i in range(len(train_set)):
        z = train_set.iloc[i, 4::7]
        v1 = train_set.iloc[i, 5::7]
        v2 = train_set.iloc[i, 6::7]
        v3 = train_set.iloc[i, 7::7]
        for j in range(len(v1)):
            v = math.sqrt(v1[j] ** 2 + v2[j] ** 2 + v3[j] ** 2) + z[j] * 10
            n.loc[i, "energy_" + str(j)] = v
    return n


def rule_based_classify(train_set, check_set):
    d = {
        (C1, C1): 0,
        (C2, C2): 0,
        (C1, C2): 0,
        (C2, C1): 0
    }
    for i, row in train_set.iterrows():
        d[(classify(row), row["class"])] += 1
    print(d)
    print((d[(C1, C1)] + d[(C2, C2)]) * 100 / sum(d.values()))
    d = {
        (C1, C1): 0,
        (C2, C2): 0,
        (C1, C2): 0,
        (C2, C1): 0
    }
    for i, row in check_set.iterrows():
        d[(classify(row), class_check[i])] += 1
    print(d)
    print((d[(C1, C1)] + d[(C2, C2)]) * 100 / sum(d.values()))


def AI_classify(train_set, check_set):
    del check_set["targetName"]
    del train_set["targetName"]
    check_set = check_set.fillna(0)
    train_set = train_set.fillna(0)
    train_class = train_set["class"]
    check_class = check_set["class"]
    del train_set["class"]
    del check_set["class"]
    train_set = get_kinetic_energy(train_set)
    check_set = get_kinetic_energy(check_set)
    clf = RandomForestClassifier(max_depth=80, n_estimators=50)
    clf.fit(train_set, train_class)
    print(clf.score(check_set, check_class))


if __name__ == "__main__":
    train_set, check_set = create_data()

    # rule_based_classify(train_set, check_set)

    AI_classify(train_set, check_set)
