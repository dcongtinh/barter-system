# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown] [markdown]
# # Data walking through
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

transac = pd.read_csv("dataset/ratebeer/transac.csv", header=None)
columns = ["GiverID", "ReceiverID", "itemID", "timestamp"]
transac.columns = columns
print(transac)

number_of_row = len(transac)
# get number of products
number_of_product = len(np.unique(transac["itemID"]))

transac_grouped = transac.groupby(["GiverID", "ReceiverID", "timestamp"]).aggregate(lambda x: str(list(np.unique(x))))
print(transac_grouped)

transac_grouped = transac_grouped.drop_duplicates()
number_of_transac = len(transac_grouped)
print(transac_grouped)

_index = [i for i in range(len(transac_grouped))]
transac_grouped.index = _index
print(transac_grouped)

for i in range(len(transac_grouped)):
    transac_grouped.iloc[i][0] = list(eval(transac_grouped.iloc[i][0]))
print(transac_grouped)

item_values = np.array([])
for i in range(len(transac_grouped)):
    item_values = np.hstack((item_values, transac_grouped.iloc[i][0]))
item_values = np.unique(item_values).astype(int)

map_col, rmap_col = {}, {}
idx, _columns = 0, []
for val in item_values:
    map_col[val] = idx
    rmap_col[idx] = val
    idx += 1

for i in range(len(transac_grouped)):
    transac_grouped.iloc[i][0] = [map_col[val] for val in transac_grouped.iloc[i][0]]
print(transac_grouped)

def transactionEncoder(df):
    # 'transactions' is now temporary variable
    transactions = [row["itemID"] for index, row in df.iterrows()]
    from mlxtend.preprocessing import TransactionEncoder

    transaction_encoder = TransactionEncoder()
    transac_matrix = transaction_encoder.fit_transform(transactions).astype("int")
    transac_df = pd.DataFrame(transac_matrix, columns=transaction_encoder.columns_)

    return transac_df, transac_matrix

transac_df, transac_matrix = transactionEncoder(transac_grouped)

print(transac_df.head())

print("Number of row:", number_of_row)
print("Number of product:", number_of_product)
print("Number of transac:", number_of_transac)

print(np.unique(transac_df.sum(), return_counts=True))


def to_row_df(items, _columns):
    row_df = pd.DataFrame(data=[np.zeros(len(_columns)).astype(int)], columns=_columns)
    for i in items:
        row_df[i] = 1
    return row_df

def predict (utility_df, utility_matrix, test_items, return_num=5):
    sim_items = cosine_similarity(utility_df, utility_matrix[:])[0]
    tu = mau = 0
    for j in test_items:
        for i in range(utility_matrix.shape[0]):
            tu += sim_items[i] * utility_matrix[i][j]
            mau += sim_items[i]
        pred = tu/mau
        if pred > 0:
            return 1
    return 0

pivot = int(0.7*len(transac_grouped))
# print(pivot)
train_set = transac_grouped[:pivot]
test_set  = transac_grouped[pivot:]
print(len(train_set))
print(len(test_set))

def in_train_lst(lst, _columns):
    for i in lst:
        if i not in _columns or i > len(_columns):
            return False
    return True

def givenN_evaluate(train, test, given_num):
    import time
    start_time = time.time()
    train_df, train_matrix = transactionEncoder(train)
    score = cnt = 0
    for i in range(len(test)):
        lst = test.iloc[i][0]
        if len(lst) <= given_num or not in_train_lst(lst, train_df.columns):
            continue
        given_items = lst[:-given_num]
        test_items = lst[-given_num:]
        row_df = to_row_df(test_items, train_df.columns)
        _start_time = time.time()
        score += predict(row_df, train_matrix, test_items, 10)
        cnt += 1
        print("{}/{} - score = {} - time = {}".format(cnt, i+1, score, time.time() - _start_time))
        if cnt == 10:
            break
    print(time.time() - start_time)
    print('cnt =', cnt)
    return score

score = givenN_evaluate(train_set, test_set, 1) #given 1
print('score =', score)


