# Practical Machine Learning - Course Project
========================================================

## Methodology
1. Divide the training set into 3 sets - 60:20:20 (train, test, test2).  
2. Use training set for training different models - **rpart, random forest, knn** and **svm**.  
3. Predict and create ConfusionMatrix on **test set** to measure performance of models and tuning parameters.
4. For the best performing model in Step 3, use test2 set **exactly once** to compute Out Of Sample Error
5. Predict using the best performing model in Step 3, classification for 20 test samples. Create the submission files

Tips: using foreach and doSNOW package to use 6 cores out of 8 cores on my windows computer


```r
library(caret) 
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(foreach)
library(doSNOW)
```

```
## Loading required package: iterators
## Loading required package: snow
```

```r
ncores <- 6
set.seed(1234)
```


```r
# load data
rawData <- read.csv("pml-training.csv", na.strings=c("", "NA", "#DIV/0!"))
# discard NAs
na.count <- apply(rawData, 2, function(x) {sum(is.na(x))}) 
validData <- rawData[, which(na.count == 0)]

# remove useless predictors
removeColumns <- grep("timestamp|X|user_name|new_window|num_window", names(validData))
validData <- validData[ ,-removeColumns]

# make training set 60
trainIndex     <- createDataPartition(y = validData$classe, p=0.60, list=FALSE)
trainData      <- validData[trainIndex, ]
nontrainData   <- validData[-trainIndex, ]
# divide test data into test and validation (20:20)
testIndex  <- createDataPartition(y = nontrainData$classe, p=0.50, list=FALSE)
testData   <- nontrainData[testIndex, ]
testData2  <- nontrainData[-testIndex, ]

classeColumnIndex <- grep("classe", names(testData))
dim(trainData); dim(testData); dim(testData2)
```

```
## [1] 11776    53
```

```
## [1] 3923   53
```

```
## [1] 3923   53
```
# Rpart model


```r
# run rpart model
system.time(
        model_rpart <- train(classe ~ ., data = trainData, method="rpart")
        )
```

```
## Loading required package: rpart
```

```
##    user  system elapsed 
##   60.56    0.20   61.23
```

```r
model_rpart
```

```
## CART 
## 
## 11776 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.04  0.5       0.4    0.03         0.04    
##   0.06  0.4       0.2    0.06         0.1     
##   0.1   0.3       0.07   0.04         0.06    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04.
```


```r
preds_rpart <- predict(model_rpart, testData[, -classeColumnIndex])
```

```
## Loading required package: rpart
```

```r
confusionMatrix(preds_rpart, testData[, classeColumnIndex])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1009  311  329  285  110
##          B   15  253   29  109  104
##          C   91  195  326  249  181
##          D    0    0    0    0    0
##          E    1    0    0    0  326
## 
## Overall Statistics
##                                         
##                Accuracy : 0.488         
##                  95% CI : (0.472, 0.504)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.331         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.904   0.3333   0.4766    0.000   0.4521
## Specificity             0.631   0.9188   0.7789    1.000   0.9997
## Pos Pred Value          0.494   0.4961   0.3129      NaN   0.9969
## Neg Pred Value          0.943   0.8517   0.8757    0.836   0.8902
## Prevalence              0.284   0.1935   0.1744    0.164   0.1838
## Detection Rate          0.257   0.0645   0.0831    0.000   0.0831
## Detection Prevalence    0.521   0.1300   0.2656    0.000   0.0834
## Balanced Accuracy       0.768   0.6261   0.6278    0.500   0.7259
```

### Rpart model is not very accurate (only 50%). 

# Random Forest


```r
# power up cluster
cl <- makeCluster(ncores)
registerDoSNOW(cl)

cv_opts <- trainControl(method="cv", number=10, allowParallel=TRUE)

# run randomforest model
system.time(
        model_rf <- train(classe ~ ., data = trainData, 
                method="rf", trControl = cv_opts)
        )
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
##    user  system elapsed 
##   48.08    0.30  459.26
```

```r
# stop cluster
stopCluster(cl)

model_rf
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 10599, 10600, 10598, 10598, 10599, 10598, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.003        0.004   
##   30    1         1      0.002        0.003   
##   50    1         1      0.004        0.005   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
preds_rf <- predict(model_rf, testData[, -classeColumnIndex])
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
confusionMatrix(preds_rf, testData[, classeColumnIndex])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    7    0    0    0
##          B    0  750    5    0    4
##          C    0    2  675    9    1
##          D    0    0    4  632    3
##          E    0    0    0    2  713
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.987, 0.993)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.988         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.988    0.987    0.983    0.989
## Specificity             0.998    0.997    0.996    0.998    0.999
## Pos Pred Value          0.994    0.988    0.983    0.989    0.997
## Neg Pred Value          1.000    0.997    0.997    0.997    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.172    0.161    0.182
## Detection Prevalence    0.286    0.193    0.175    0.163    0.182
## Balanced Accuracy       0.999    0.993    0.992    0.990    0.994
```
### Random Forest is highly accurate (99.1%)

# KNN Model


```r
# power up cluster
cl <- makeCluster(ncores)
registerDoSNOW(cl) 

# run knn model
knn_opts <- data.frame(.k=c(seq(3, 11, 2)))

system.time(
        model_knn <- train(classe ~ ., data = trainData, 
                method="knn",  trControl = cv_opts, tuneGrid = knn_opts)
        )
```

```
##    user  system elapsed 
##    0.78    0.10   54.41
```

```r
# stop cluster
stopCluster(cl)

model_knn
```

```
## k-Nearest Neighbors 
## 
## 11776 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 10598, 10598, 10599, 10600, 10599, 10599, ... 
## 
## Resampling results across tuning parameters:
## 
##   k   Accuracy  Kappa  Accuracy SD  Kappa SD
##   3   0.9       0.9    0.008        0.01    
##   5   0.9       0.9    0.01         0.02    
##   7   0.9       0.8    0.01         0.02    
##   9   0.8       0.8    0.01         0.02    
##   10  0.8       0.8    0.01         0.01    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 3.
```


```r
preds_knn <- predict(model_knn, testData[, -classeColumnIndex])
confusionMatrix(preds_knn, testData[, classeColumnIndex])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1086   26    5    8    7
##          B    4  673   19    3   25
##          C    6   25  630   32   17
##          D   15   19   20  594   17
##          E    5   16   10    6  655
## 
## Overall Statistics
##                                         
##                Accuracy : 0.927         
##                  95% CI : (0.919, 0.935)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.908         
##  Mcnemar's Test P-Value : 5.05e-06      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.973    0.887    0.921    0.924    0.908
## Specificity             0.984    0.984    0.975    0.978    0.988
## Pos Pred Value          0.959    0.930    0.887    0.893    0.947
## Neg Pred Value          0.989    0.973    0.983    0.985    0.980
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.277    0.172    0.161    0.151    0.167
## Detection Prevalence    0.289    0.185    0.181    0.170    0.176
## Balanced Accuracy       0.978    0.935    0.948    0.951    0.948
```
### KNN is about 92.7% accurate, which is quite good and this was much faster to train than random forest.

# SVM model


```r
# power up cluster
cl <- makeCluster(ncores)
registerDoSNOW(cl)

# run knn model

system.time(
        model_svm <- train(classe ~ ., data = trainData, 
                method="svmLinear",  trControl = cv_opts, tuneLength = 5)
        )
```

```
## Loading required package: kernlab
```

```
##    user  system elapsed 
##   15.57    0.33   75.76
```

```r
# stop cluster
stopCluster(cl)

model_svm
```

```
## Support Vector Machines with Linear Kernel 
## 
## 11776 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 10597, 10598, 10598, 10598, 10600, 10599, ... 
## 
## Resampling results
## 
##   Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.8       0.7    0.02         0.02    
## 
## Tuning parameter 'C' was held constant at a value of 1
## 
```


```r
preds_svm <- predict(model_svm, testData[, -classeColumnIndex])
```

```
## Loading required package: kernlab
```

```r
confusionMatrix(preds_svm, testData[, classeColumnIndex])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1046  100   75   48   30
##          B   14  548   64   32  101
##          C   30   48  516   71   51
##          D   24    8   19  454   38
##          E    2   55   10   38  501
## 
## Overall Statistics
##                                         
##                Accuracy : 0.781         
##                  95% CI : (0.768, 0.794)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.722         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.937    0.722    0.754    0.706    0.695
## Specificity             0.910    0.933    0.938    0.973    0.967
## Pos Pred Value          0.805    0.722    0.721    0.836    0.827
## Neg Pred Value          0.973    0.933    0.948    0.944    0.934
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.267    0.140    0.132    0.116    0.128
## Detection Prevalence    0.331    0.193    0.183    0.138    0.154
## Balanced Accuracy       0.924    0.828    0.846    0.839    0.831
```
### SVM was only 78.1% accurate. Which is not as good as others.

# Results
### We have the best prediction from Random Forest, so we select Random Forest as our final model.
### First we calculate out of sample error on testData2 (which was havent used till now)

```r
oos_rf <- predict(model_rf, testData2[, -classeColumnIndex])
confusionMatrix(oos_rf, testData2[, classeColumnIndex])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116   11    0    0    0
##          B    0  744    9    1    0
##          C    0    4  671   11    1
##          D    0    0    4  631    1
##          E    0    0    0    0  719
## 
## Overall Statistics
##                                         
##                Accuracy : 0.989         
##                  95% CI : (0.986, 0.992)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.986         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.980    0.981    0.981    0.997
## Specificity             0.996    0.997    0.995    0.998    1.000
## Pos Pred Value          0.990    0.987    0.977    0.992    1.000
## Neg Pred Value          1.000    0.995    0.996    0.996    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.190    0.171    0.161    0.183
## Detection Prevalence    0.287    0.192    0.175    0.162    0.183
## Balanced Accuracy       0.998    0.989    0.988    0.990    0.999
```
### Accuracy is 0.989.  
### Out of Sample Error is 0.011.  
### 95% CI is (0.008 to 0.014).  

# Predicting outcome on given 20 test cases

```r
# Predict the outcomes in the given test file
classetestData <- read.csv("pml-testing.csv", na.strings=c("", "NA", "#DIV/0!"))
removeColumns <- grep("timestamp|X|user_name|new_window|num_window", names(classetestData))
classetestData <- classetestData[ ,-removeColumns]
answers <- predict(model_rf, classetestData)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# write output to files

pml_write_files <- function(answers){
        x <- as.character(answers)
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

pml_write_files(answers)
```
