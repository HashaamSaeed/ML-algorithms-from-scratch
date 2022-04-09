import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
      self.data = data
      opt_dict = {}   ## { ||w||: [w,b] } ## mag of w will be key and values will be w,b

      transforms = [[1,1],   ## Finally, we set our transforms. We've explained that our intention there is to make sure we check every version of the vector possible.
                    [-1,1],
                    [-1,-1],
                    [1,-1]]

      ## Next, we need some starting point that matches our data. To do this, we're going to first reference our training data to pick some haflway decent starting values:
      all_data = []
      for yi in self.data:
          for featureset in self.data[yi]:
              for feature in featureset:
                  all_data.append(feature)

      self.max_feature_value = max(all_data)
      self.min_feature_value = min(all_data)
      all_data = None

      # support vectors yi(xi.w+b) = 1

      ## All we're doing here is cycling through all of our data, and finding the highest and lowest values. Now we're going to work on our step sizes:
       # What we're doing here is setting some sizes per step that we want to make. For our first pass, we'll take big steps (10%). Once we find the minimum with these steps,
       # we're going to step down to a 1% step size to continue finding the minimum here. Then, one more time, we step down to 0.1% for fine tuning. We could continue stepping down,
       # depending on how precise you want to get. I will discuss towards the end of this project how you could determine within your program whether or not you should continue 
       # optimizing or not.
      step_sizes = [self.max_feature_value * 0.1,
                    self.max_feature_value * 0.01,
                    # point of expense:
                    self.max_feature_value * 0.001,]

      ## Next, we're going to set some variables that will help us make steps with b (used to make larger steps than we use for w, since we care far more about w precision than b), and keep track of the latest optimal value:
      # extremely expensive
      b_range_multiple = 5
      # 
      b_multiple = 5
      latest_optimum = self.max_feature_value*10


      ## The idea here is to begin stepping down the vector. To begin, we'll set optimized to False, and we'll reset this for each major step. The optimized var will be true when we have checked all steps down to the base of the convex shape (our bowl).
      for step in step_sizes:
          w = np.array([latest_optimum,latest_optimum])
          # we can do this because convex
          optimized = False
          while not optimized:
            for b in np.arange(-1*(self.max_feature_value*b_range_multiple),self.max_feature_value*b_range_multiple,step*b_multiple):
              ## Here, we begin also iterating through possible b values, and now you can see our b values we set earlier in action. I will note here that we're straight iterating through 
               # b with a constant step-size. We could also break down b steps just like we did with w. To make things more accurate and precise, you probably would want to implement that. 
               # That said, I am going to skip doing that for brevity, since we'll achieve similar results either way and we're not trying to win any awards here


              for transformation in transforms:
                w_t = w*transformation
                found_option = True
                # weakest link in the SVM fundamentally
                # SMO attempts to fix this a bit
                # yi(xi.w+b) >= 1
                # 
                # #### add a break here later..
                for i in self.data:
                    for xi in self.data[i]:
                        yi=i
                        if not yi*(np.dot(w_t,xi)+b) >= 1:
                            found_option = False
                            
                if found_option:
                    opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                    ## Now we iterate through each of the transformations, testing each of them against our constraint requirements. If any of the featuresets within our data don't meet 
                     # our constraints, then we toss the variables as they don't fit and we move on. I commented in a suggestion for a break here. If just one variable doesn't work you might 
                     # as well give up on the rest since just 1 that doesn't fit is enough to toss the values for w and b. You could break there, as well as in the preceeding for loop. For now,
                     #  I will leave the code as I originally had it, but I thought of the change whenever I was filming the video version.


            if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
            else:
              w = w - step


            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2


              ## Once we've passed zero with our stepping of the w vector, there's no reason to continue since we've tested the negatives via the transformation, 
               # thus we'll be done with that step size and either continue to the next, or be done entirely. If we've not passed 0, then we take another step. Once we've taken 
               # all of the steps that we want to take, then we're going to sort a list of all dictionary keys from our opt_dict (which contains ||w|| : [w,b]). We want the smallest
               # magnitude of vector w, so we go with the first item in that list. We set self.w and self.b from here, we set latest optimums, and we either may take another step or 
               # be totally done with the entire process (if we have no more steps to take).



    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)  ## sign(x.w+b)
        ## numpy.sign(array [, out]) function is used to indicate the sign of a number element-wise. For integer inputs, if array value is greater 
         # than 0 it returns 1, if array value is less than 0 it returns -1, and if array value 0 it returns 

        if classification != 0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*', c=self.colors[classification])
        else:
            print('featureset',features,'is on the decision boundary')

        return classification

    def visualize(self):
        #scattering known featuresets.
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()



## -1 and 1 are two classes and are list of lists of features/points         
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}



svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()             