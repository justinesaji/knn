from scipy.spatial import distance #scipy mod class spatial distance func 

def eucli(a,b): #calc eucli distance
	return distance.euclidean(a,b)

class myknn():
  def fit(self,x_train,y_train):
    self.x_train=x_train
    self.y_train=y_train
  def predict(self,x_test):
    predictions=[]
    for row in x_test:
        labels=self.closest(row)
        predictions.append(labels)
    return predictions
  def closest(self,row): #to calc nearest dist
    best_dist=eucli(row,self.x_train[0])
    best_index=0
    for i in range(1,len(self.x_train)):
        dist=eucli(row,self.x_train[i])
        if dist<best_dist:
            best_dist=dist
            best_index=i
    return self.y_train[best_index]

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()
x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

clf=myknn()
clf.fit(x_train,y_train)

p=clf.predict(x_test)

print("accuracy=",accuracy_score(y_test,p))
# load iris the datasets
 #dataset = datasets.load_iris()
# fit a k-nearest neighbor model to the data
 #model = KNeighborsClassifier()
#model.fit(dataset.data, dataset.target)
#print(model)
# make predictions
#expected = dataset.target
#predicted = model.predict(dataset.data)
# summarize the fit of the model
#print("accuracy=",accuracy_score(expected,predicted))
