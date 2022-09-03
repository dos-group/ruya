import pandas as pd, numpy as np

from sklearn.linear_model import LinearRegression as LR

# Memory usage estimation based on local profiling runs
# Extended from Crispy (https://github.com/dos-group/crispy)

df_cluster = pd.read_csv('arrow_cluster_jobs.csv')
df_local = pd.read_csv('local_jobs.csv')

jobs = list(sorted(set(df_cluster['job'])))

def get_train_data(job):
    df_train = df_local[df_local['job'] == job]
    X_train = df_train['dataset_size'].to_numpy().reshape(-1,1)
    y_train = df_train['max_memory_used'].to_numpy()

    df_test = df_cluster[df_cluster['job'] == job]
    X_test = [[df_test.iloc[0]['input_size']]]

    return X_train, y_train, X_test


def get_memory_requirement(job):
    X_train, y_train, X_test = get_train_data(job)
    model = LR()
    model.fit(X_train,y_train)
    score = model.score(X_train, y_train)
    if score > .99: return model.predict(X_test)[0]//1e9
    elif score > .1: return -1
    else: return 0
