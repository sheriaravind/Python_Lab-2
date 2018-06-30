from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC

def SVM_Demo():
    iris_data = datasets.load_iris()
    a = iris_data.data
    b = iris_data.target
    atrain, atest, btrain, btest = train_test_split(a, b, test_size=0.2,
                                                    random_state=20)  # Data split 20% test data and 80% Training data
    linear_kernel = SVC(kernel='linear')  # Linear kernel
    rbf_kernel = SVC(kernel='rbf')  # rbf Kernel
    linear_kernel.fit(atrain, btrain)  # Data fit into the linear kernel
    b_linear_pred_value = linear_kernel.predict(atest)  # Predicted value using linear kernel
    rbf_kernel.fit(atrain, btrain)  # Data fit into the rbf kernel
    b_rbf_pred_value = rbf_kernel.predict(atest)  # Predicted value using
    print("Accuracy Score/Linear Kernal : ",
          accuracy_score(b_linear_pred_value, btest))  # Calculating the accuracy score
    print("Accuracy Score/RBF Kernal : ", accuracy_score(b_rbf_pred_value, btest))

if __name__ == '__main__':
    SVM_Demo()