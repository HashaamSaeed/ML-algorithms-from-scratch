import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result,confidence 

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()  ## converts whole dataframe to float cuz of the ? problem 

random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]  ## 0-80% of data
test_data = full_data[-int(test_size*len(full_data)):]   ## 80-100% of data


## populating our dicts 
## Now we populate the dictionaries. If it is not clear, the dictionaries have two keys: 2 and 4. The 2 is for the benign tumors (the same value the actual dataset used), 
 # and the 4 is for malignant tumors, same as the data. We're hard coding this, but one could take the classification column, and create a dictionary like this with keys 
 # that were assigned by unique column values from the class column. We're just going to keep it simple for now, however.

for i in train_data:
    train_set[i[-1]].append(i[:-1])  ## basically here we take the train_data last colum and that becomes i which is our class and then we append on that value either being 2 or 4 the entire data that it belongs to till -1 
                                      # because last column is class in the dataset

for i in test_data:
    test_set[i[-1]].append(i[:-1])




correct = 0
total = 0


## Now, we first iterate through the groups (the classes, 2 or 4, also the keys in the dictionary) in the test set, then we go through each datapoint, 
 # feeding that datapoint through to our custom k_nearest_neighbors, along with our training data, train_set, and then our choice for k, which is 5. 
 # I chose 5 purely because that's the default for the Scikit Learn KNeighborsClassifier

for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total, confidence )



## Finally, one last point I will make is on actual prediction confidence. There are two ways to measure confidence. One way is by comparing how many 
 # examples you got correct vs incorrect in the testing stage, but, another way, is by checking the vote percentage. For example, your overall algorithm may
 # be 97% accurate, but on some of the classifications the votes may have been 3 to 2. While 3 is the majority, it's only 60% vote rather than 100% which would be ideal. 
 # In terms of telling someone whether or not they have breast cancer, like the automatic car differentiating between a blob of tar and a child in a blanket, you probably 
 # prefer 100%! Thus, in the case of 60% vote on a 97% accurate classifier, you can be 97% sure that you are only 60% certain about your classification. It's totally possible 
 # that this 60% vote is responsible for part of the 3% that was inaccurately measured.