# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %%
# # Data walking through
import pandas as pd
import numpy as np

# read csv and assign columns
columns = ["personId", "productId"]
df = pd.read_csv("dataset/ratebeer/wish.csv", header=None)
df.columns = columns

number_of_row = len(df)
# get number of products
number_of_product = len(np.unique(df["productId"]))
# get number of person
number_of_person = len(np.unique(df["personId"]))
df = df.groupby("personId").aggregate(lambda x: list(np.unique(x)))
print(df)

columns = ["personId", "productId"]
have_df = pd.read_csv("dataset/ratebeer/have.csv", header=None)
have_df.columns = columns
have_df = have_df.groupby("personId").aggregate(lambda x: list(np.unique(x)))
print(have_df)

_df = df
cnt = 0
for i in range(len(have_df)):
    have_PID = have_df.iloc[i].name
    if have_PID in _df.index.values:
        _df.loc[have_PID][0] = np.append(_df.loc[have_PID][0], have_df.iloc[i][0])
        _df.loc[have_PID][0] = np.unique(_df.loc[have_PID][0])
    else:
        _df = _df.append(have_df.iloc[i])
print(_df)

for i in range(len(_df)):
    _df.iloc[i][0] = str(list(_df.iloc[i][0]))

_df = _df.drop_duplicates()
print(_df)

_index = [i for i in range(len(_df))]
_df.index = _index
print(_df)

for i in range(len(_df)):
    _df.iloc[i][0] = list(eval(_df.iloc[i][0]))
print(_df)

item_values = np.array([])
for i in range(len(_df)):
    item_values = np.hstack((item_values, _df.iloc[i][0]))
item_values = np.unique(item_values).astype(int)
print(item_values)

map_col, rmap_col = {}, {}
idx, _columns = 0, []
for val in item_values:
    map_col[val] = idx
    rmap_col[idx] = val
    _columns.append(idx)
    idx += 1

for i in range(len(_df)):
    _df.iloc[i][0] = [map_col[val] for val in _df.iloc[i][0]]
print(_df)

def transactionEncoder(df):
    # 'transactions' is now temporary variable
    transactions = [row["productId"] for index, row in df.iterrows()]
    from mlxtend.preprocessing import TransactionEncoder

    transaction_encoder = TransactionEncoder()
    wish_matrix = transaction_encoder.fit_transform(transactions).astype("int")
    wish_df = pd.DataFrame(wish_matrix, columns=transaction_encoder.columns_)

    return wish_df, wish_matrix

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

wish_df, wish_matrix = transactionEncoder(_df)

print(wish_df.head())

print(np.unique(wish_df.sum(), return_counts=True))
# # Transaction-based wish prediction using apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(wish_df, min_support=0.2)
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="confidence")
# Something FAILED
print(rules)
# # User-based prediction using cosine similarity
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

def to_row_df(items, _columns):
    row_df = pd.DataFrame(data=[np.zeros(len(_columns)).astype(int)], columns=_columns)
    for i in items:
        row_df[i] = 1
    return row_df

def predict (utility_df, utility_matrix, test_items, return_num=5):
    sim_items = cosine_similarity(utility_df, utility_matrix[:])[0]

    for j in test_items:
        tu = mau = 0
        for i in range(utility_matrix.shape[0]):
            tu += sim_items[i] * utility_matrix[i][j]
            mau += sim_items[i]
        pred = tu/mau
        if pred > 0:
            return 1
    return 0

pivot = int(0.7*len(_df))
# print(pivot)
train_set = _df[:pivot]
test_set  = _df[pivot:]
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
        # if cnt == 10:
        #     break
    print(time.time() - start_time)
    print('cnt =', cnt)
    return score

score = givenN_evaluate(train_set, test_set, 1) #given 1
print('score =', score)


