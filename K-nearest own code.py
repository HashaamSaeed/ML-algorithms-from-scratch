import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')



def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:  ## data here will be passed as a dict and the length of that is the number of keys 
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))  ## linalg = linear algebra 
            distances.append([euclidean_distance,group]) ## will create a list of lists with distance and group tuples

    print(distances)
    votes = [i[1] for i in sorted(distances)[:k]]  ## i[1] corresponds to the group ## populates vote with distances 
    ## Above, we're going through each list, within the distance list, which is sorted. The sort method will sort based on the first item in each list within the list.
     # That first item is the distance, so when we do sorted(distances), we're sorting the list by shortest to greatest distance. Then we take the [:k] of this list, 
     # since we're interest in only k of the nearest neighbors. Finally, in the outermost layer of this for loop, we're taking i[0], where i was the list within the list, 
     # which contained [distance,class]. Beyond finding the k closest distances, we actually do not care what the distance was in this case, just the class now.
    print(votes)

    vote_result = Counter(votes).most_common(1)[0][0]
    ## Collections finds the most common elements. In our case, we just want the single most common, but you can find the top 3 or top 'x.
     # ' Without doing the [0][0] part, you get [('r', 3)]. Thus, [0][0] gives us the first element in the tuple. The three you see is how many votes 'r' got.
    

    return vote_result    



dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}  ## 2 clusters k,r and their points 
new_features = [5,7]  ## a new point 

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]  ## s=100 size of the dot 

#same code as above 
"""
for i in dataset:  ## i is k,r
    for ii in dataset[i]:  ## ii is point set [1,2]
        plt.scatter(ii[0],ii[1],s=100,color=i)  ## plot settings 
"""

plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.show()





        

