# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %%'
# # Data walking through

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

transac = pd.read_csv("dataset/ratebeer/transac.csv", header=None)
columns = ["GiverID", "ReceiverID", "itemID", "timestamp"]
transac.columns = columns
transac


number_of_row = len(transac)
# get number of products
number_of_product = len(np.unique(transac["itemID"]))

transac_grouped = transac.groupby(["GiverID", "ReceiverID", "timestamp"]).aggregate(lambda x: list(np.unique(x)))
number_of_transac = len(transac_grouped)


_index = [i for i in range(len(transac_grouped))]
transac_grouped.index = _index
transac_grouped


print("Number of row:", number_of_row)
print("Number of product:", number_of_product)
print("Number of transac:", number_of_transac)


def to_row_df(items, _columns):
    row_df = pd.DataFrame(data=[np.zeros(len(_columns)).astype(int)], columns=_columns)
    for i in items:
        row_df[i] = 1
    return row_df

def predict (utility_df, utility_matrix, return_num=5):
    for i in range(len(utility_matrix)):
        sim = cosine_similarity(utility_df, utility_matrix[i:i+1])[0][0]
        if 0. < sim and sim < 1.:
            return transac_grouped.iloc[i][0]
    return []

def transactionEncoder(df):
    # 'transactions' is now temporary variable
    transactions = [row["itemID"] for index, row in df.iterrows()]
    from mlxtend.preprocessing import TransactionEncoder

    transaction_encoder = TransactionEncoder()
    transac_matrix = transaction_encoder.fit_transform(transactions).astype("int")
    transac_df = pd.DataFrame(transac_matrix, columns=transaction_encoder.columns_)

    return transac_df, transac_matrix

pivot = int(0.7*len(transac_grouped))
train_set = transac_grouped[:pivot]
test_set  = transac_grouped[pivot:]
print('train_set=', len(train_set))
print('test_set=', len(test_set))


def in_train_lst(lst, _columns):
    for i in lst:
        if i not in _columns:
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
        suggests = predict(row_df, train_matrix)
        print("{}/{} {} - {}".format(cnt+1, i+1, lst[given_num:], suggests))
        if len(suggests):
            for s in suggests:
                if s in test_items:
                    score += 1
                    break
        cnt += 1

    print(time.time() - start_time)
    print('cnt=', cnt)
    return score, score/len(test)

score, mean = givenN_evaluate(train_set, test_set, 1) #given 1
print(score, mean)


