data_scores = [
    [75, 20, -84, -5, 24], #beg, year
    [109, 105, 83, 131, 142], # int, week
    [147, 122, 106, 150, 150], # adv, dail
    [9, 55, 15, 75, 100], # beg, year
    [100, 65, -8, 100, 65], # noe, neve
    [125, 120, 125, 120, 150], # beg, year
    [125, 75, 150, 125, 150], # noe, neve
    [109, 43, 32, 47, 107], # int, mont
    [100, 75, 125, 90, 150], # exp, week
    [100, 62, 75, 95, 125], # beg, mont
    [150, 142, 110, 100, 150], # adv, dail
    [150, 125, 108, 150, 150], # int, year
    [71, 30, 115, 95, 150], # beg, year
    [63, 70, 115, 68, 150], # exp, year
    [142, 106, 76, 133, 143], # int, week
    [50, 14, 90, 90, 125], # int, mont
    [64, 16, 123, 110, 122], # Beg, year
    [45, 112, 137, 125, 150], # adv, mont
    [109, 121, 140, 150, 121], # adv, mont
    [125, 85, 82, 150, 150] # int, week
]


import pandas as pd
import os
import numpy as np
data_scores = []
for idx, user in enumerate(os.listdir(DATA_FOLDER)):
    sc=[]
    for trial in range(3, 8):
        file = get_user_path(user, trial)
        df = pd.read_csv(file)
        all_scores = [int(k.split(' ' )[0]) for k in df['Score']]
        all_scores = np.array(all_scores)
        score_dif = all_scores[1:] - all_scores[:-1]
        score = sum(score_dif[score_dif<0]) + 25*data[idx][trial-3]
        sc.append(score)
    data_scores.append(sc)
