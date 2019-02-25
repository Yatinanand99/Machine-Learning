#Importing Important Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("data.csv")

X = dataset.iloc[:,54:88].values

#Handling the missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)

#Clustering the data with optimum no. of Cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 2,init='k-means++',n_init=10,max_iter=300,random_state=0)

#Predecting and Plotting the data
def __plot__(X_plot, label):
    y_pred=kmeans.fit_predict(X_plot)
    plt.scatter(X_plot[y_pred==0,0],X_plot[y_pred==0,1],s=4,c='red', label='Cluster 1')
    plt.scatter(X_plot[y_pred==1,0],X_plot[y_pred==1,1],s=4,c='blue', label='Cluster 2')
    plt.savefig('Images/'+label+'.png')
 

#Fining the dataset for different categories    
X_goalkeeper = X[:,-5:] # Goal-Keeper
X_attacker = X[:,0:10]
X_overall = X[:,[0,1,3,4,5,6,7,9,10,11,12,13,14,15,17,18,19,22,23,24]]

__plot__(X_goalkeeper, "goal_keeper")
__plot__(X_attacker, "attacker")
__plot__(X_overall, "overall")