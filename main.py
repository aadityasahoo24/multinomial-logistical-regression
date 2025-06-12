import pandas as pd
import numpy as np
#from sklearn.metrics import confusion_matrix

nc = 5 #no. of classes

df = pd.read_csv("seattle-weather.csv")
df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)

df['day_sin'] = np.sin(2*np.pi*df['day']/30)
df['day_cos'] = np.cos(2*np.pi*df['day']/30)

y = df['weather']
y = y.map({"rain": 1, "sun": 2, "fog": 3, "drizzle": 4, "snow": 0})
#list of [0, 1, 1 .. 2, 4, 1] etc. 

df = df.drop(columns=['date', 'weather'])
X = df.values

m, n = X.shape #m = no. of samples, n = no. of features

#convert y to one-hot values
#remember that y needs to be mapped like 0 1 2 instead of 1 2 3 for this onehot to work
Y = np.zeros((m, nc)) #creates m (rows) number of lists of 0s of length nc (columns)
Y[np.arange(m), y] = 1 #for each ith example, select y[i]th column and set to 1

perm = np.random.permutation(len(X))

X = X[perm]
Y = Y[perm]

train_size = int(0.8 * len(X))
X, X_test = X[:train_size], X[train_size:]
Y, Y_test = Y[:train_size], Y[train_size:]
#other methods to encode: 
'''
from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse_output=False)
Y = en.fit_transform(y.values.reshape(-1, 1))
'''

theta = np.zeros((n, nc))

#print(X.shape, theta.shape, Y.shape) #size checking 

#use sigmoid for binary
def softmax(z): #for multilogistical classification
    ''' using raw z value leads to overflow of softmax.
    out = [(np.exp(i)/np.sum(np.exp(z))) for i in z]
    
    #use shifted z = z[i] - z_max
    out = [(np.exp(i-np.max(z))/np.sum(np.exp(z-np.max(z)))) for i in z]
    return np.array(out, dtype='float64')
    '''
    #this approach may still not work for higher order lists
    z = np.array(z)
    z_max = np.max(z, axis = 1, keepdims = True)
    exp_z = np.exp(z - z_max)
    return exp_z/np.sum(exp_z, axis = 1, keepdims = True)

#print(np.sum(softmax([3, 1, 2]))) #returns 1 -> works as required

def grad_log(theta):
    h = softmax(X.dot(theta)) #run h = g(x.theta)
    l = Y * np.log(h + 1e-15) #adding small value so log(0) doesnt give error
    ''' Alternate method:
    for i in range(m):
        g += np.outer(X[i], (Y[i] - h[i])) #outer cause were doing matmul in loop
    return g
    '''
    g = X.T.dot(Y - h) 
    return [g, l]

alpha = 0.025
epochs = 10000

for e in range(epochs):
    g, l = grad_log(theta)
    theta += alpha/m * g
    #if e%100 == 0:
        #print(f"Epoch: {e}, log-likelyhood: {np.sum(l)}")



def predict(X, theta):
    h = softmax(X @ theta)
    return np.argmax(h, axis = 1)

prediction = predict(X_test, theta)
results = np.argmax(Y_test, axis=1)
print(np.mean(prediction == results))

#cm = confusion_matrix(results, prediction)
#print(cm)
