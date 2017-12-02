#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:33:10 2017

@author: sabarnikundu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:27:18 2017

@author: sabarnikundu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:06:27 2017

@author: sabarnikundu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:54:03 2017

@author: sabarnikundu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:34:31 2017

@author: sabarnikundu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:17:18 2017

@author: sabarnikundu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
Y= dataset.iloc[:,4].values

from sklearn.cross_validation import train_test_split
X_train , X_test, Y_train , Y_test= train_test_split(X,Y ,test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

#fitting  the classifier
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10 ,criterion= 'entropy', random_state=0)
classifier.fit(X_train, Y_train)

#predicting new values
y_pred= classifier.predict(X_test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, y_pred)


#visualising the dataset
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train ,Y_train
X1,X2 = np.meshgrid (np.arange(start = X_set[:, 0].min() -1, stop = X_set[:,0].max()+1 , step = 0.01),
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:,1].max()+1 , step = 0.01))

plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap= ListedColormap(('red' , 'green')))
plt.xlim (X1.min(), X1.max())
plt.ylim (X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red' , 'green'))(i), label=j)
plt.title(' random forest Classifier')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.show()

