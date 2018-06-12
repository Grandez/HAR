library(caret)
library(e1071)
library(randomForest)

ds.training = 'pml-training.csv'
ds.testing  = 'pml-testing.csv'
MIN.NA <- .75

setwd('P:/R/PML')  # Set the current environment

# First load of file
pml.base = read.csv(ds.training)
head(pml.base)   # Omitted for HTML page
dim(pml.base)

NAS=c("", "NA", "#DIV/0!")
pml.base = read.csv('pml-training.csv', na.string=NAS)
prc.na = sum(is.na(pml.base)) / prod(dim(pml.base)) # check number of NA

# Calculate number of NA by column
prc.na <- colSums(sapply(pml.base, is.na)) / nrow(pml.base)

# Remove columns
cols.ignore <- names(prc.na[prc.na > MIN.NA])
pml.work <- pml.base[ , !(names(pml.base) %in% cols.ignore)]
prc.na <- sum(is.na(pml.work)) / prod(dim(pml.work))

pml.base = pml.work[, -c(1:6)]

# Create train and test files
v.train <- createDataPartition(pml.base$classe, p = 0.75, list = FALSE)
pml.train <- pml.base[v.train,]
pml.test  <- pml.base[-v.train,]

## Model 1: Support Vector Machine
model.svm <- svm(classe~., data = pml.train)
pred.svm <- predict(model.svm, newdata = pml.test)
cm.svm = confusionMatrix(pred.svm, pml.test$classe)

## Model 2: Random Forest
model.rf <- randomForest(classe~., data = pml.train, n.trees = 500)
pred.rf <- predict(model.rf, newdata = pml.test)
cm.rf = confusionMatrix(pred.rf, pml.test$classe)

## Model 3: K-nearest neighbors
train.ctrl <- trainControl(method = "boot632", number = 5)
model.knn <- train(classe~., data = pml.train, method = "knn",
                   trControl = train.ctrl,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)
pred.knn <- predict (model.knn, newdata = pml.test)
cm.knn = confusionMatrix(pred.knn, pml.test$classe)

## Results
df.res = rbind(cm.svm$overall, cm.rf$overall, cm.knn$overall)
df.res = cbind(c("SVM", "RF", "KNN"), df.res)
print(df.res[,c(1,2,4,5)])

### Random Forest - Confussion Matrix
p <- ggplot(data=as.data.frame(cm.rf$table), aes(x=Reference, y=Prediction))
p = p + geom_tile(aes(fill = log(Freq)), colour = "white")
p = p + scale_fill_gradient(low = "white", high = "steelblue")
p = p + geom_text(aes(x = Reference, y = Prediction, label = Freq))
p = p + theme(legend.position = "none")
p

### Random Forest - Sensitivity
cm.rf$byClass[,c(1,2,11)]

# Prediction

# Create pml.test according pml.train
pml.test.base =  read.csv(ds.testing, na.string=NAS)
pml.test =  pml.test.base[ , names(pml.test.base) %in% colnames(pml.train)]
pred.test <- predict(model.rf, newdata = pml.test)
print(pred.test)
