import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def LDA_Model():
    iris = datasets.load_iris() # get the data iris from the sklearn datasets
    A = iris.data #assigning the data to A
    B = iris.target #assigning the target to B
    target_names = iris.target_names
    A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.20) # Assigning the test and train data
    cf = KNeighborsClassifier(n_neighbors=3) # Defining KNeighbourClassifier
    cf.fit(A_train, B_train) # Fit the train data to the model
    B_pred = cf.predict(A_test) # Predict the value using the model.
    l = LinearDiscriminantAnalysis(n_components=3) # Defining the model Linear Discriminant Analysis
    A_R = l.fit(A_test, B_pred).transform(A)
    colors = ['green', 'blue', 'orange']
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(A_R[B == i, 0], A_R[B == i, 1], alpha=1, color=color, label=target_name) # Scattering the data
    plt.legend(loc='best', shadow=False, scatterpoints=1) #places a legend in the axes
    plt.show() # Show the scattered points on the graph

if __name__ == '__main__':
    LDA_Model()