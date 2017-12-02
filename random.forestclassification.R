dataset=read.csv('Social_Network_Ads.csv')
dataset=dataset[ ,3:5]

#factoring the target feature
dataset$Purchased=factor(dataset$Purchased, levels = c(0,1))


#diving the dataset
library(caTools)
set.seed(1234)
split=sample.split(dataset$Purchased,SplitRatio = 0.75)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#feature scaling
training_set[-3]=scale(training_set[-3])
test_set[-3]=scale(test_set[-3])

#classifier
library(randomForest)
classifier=randomForest(x=training_set[-3],
                        y=training_set$Purchased)

#predicting the results
y_pred=predict(classifier, newdata=test_set[-3])

#table
cm=table(test_set[ ,3],y_pred)

#visualising the training set
library(ElemStatLearn)
set=training_set
X1=seq(min(set[, 1]) -1, max(set[, 1]) +1 , by=0.01)
X2=seq(min(set[, 2]) -1, max(set[, 2]) +1 , by=0.01)
grid_set=expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)

plot(set[ ,-3],
     main=' random forest (training set)',
     xlab='Age' ,ylab='EstimatedSalary',
     xlim=range(X1),ylim=range(X2))
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set,pch='.', col=ifelse(y_grid==1, 'springgreen3', 'tomato'))
points(set,pch=21, bg=ifelse(set[,3]==1, 'green4', 'red3'))

