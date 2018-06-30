from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def k_Nearest():
    irisdataset = datasets.load_iris()  # Get the iris data from sklearn
    x = irisdataset.data  # Assigning data to x
    y = irisdataset.target  # Assigning target to y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # creating train and test data
    k_range = range(1, 70) # setting the K range
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k) # Assigning the k value to the neighbors for the model
        knn.fit(x_train, y_train) # Fitting the data
        y_pred = knn.predict(x_test) # Predict the value y
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred)) # Accuracy for the predicted value
        scores.append(metrics.accuracy_score(y_test, y_pred)) # Appending the accuracy values to the scores to plot the graph

    plt.plot(k_range, scores)
    plt.xlabel("K value")
    plt.ylabel("Accuracy/Testing")
    plt.show()

if __name__ == '__main__':
    k_Nearest()