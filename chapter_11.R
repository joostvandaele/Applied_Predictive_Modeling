# Chapter 11 Applied Predictive Modeling
if (!require("pacman")) install.packages("pacman")
pacman::p_load(AppliedPredictiveModeling)
set.seed(975)
simulatedTrain <- quadBoundaryFunc(500)
simulatedTest <- quadBoundaryFunc(1000)
head(simulatedTrain)

# Fit random forest and quadratic discriminant models
pacman::p_load(randomForest, MASS) # MASS for qda() function
rfModel <- randomForest(class ~ X1 + X2, 
                        data = simulatedTrain,
                        ntree = 2000)
qdaModel <- qda(class ~ X1 + X2, data = simulatedTrain)
class(rfModel)
class(qdaModel)

# predict classes
qdaTrainPred <- predict(qdaModel, simulatedTrain)
names(qdaTrainPred)
head(qdaTrainPred$class)
head(qdaTrainPred$posterior)
qdaTestPred <- predict(qdaModel, simulatedTest)
simulatedTrain$QDAprob <- qdaTrainPred$posterior[, "Class1"]
simulatedTest$QDAprob <- qdaTestPred$posterior[, "Class1"]

rfTestPred <- predict(rfModel, simulatedTest, type = "prob")
head(rfTestPred)
simulatedTest$RFprob <- rfTestPred[, "Class1"]
simulatedTest$RFclass <- predict(rfModel, simulatedTest)

# Compute sensitivity and specificity
# Class 1 will be used as the event of interest
pacman::p_load(caret) # used for computation of sensitivity and specificity
sensitivity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            positive = "Class1")
specificity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            negative = "Class2")
# Predictive values can also be computed either by using the prevalence found in the data set 
# (46%) or by using prior judgement
posPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class1")
negPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class2")
# Change the prevalence manually 
posPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class1",
             prevalence = 0.9)

# Confusion Matrix
pacman::p_load(e1071)
confusionMatrix(data = simulatedTest$RFclass,
                reference = simulatedTest$class,
                positive = "Class1")

# Receiver Operating Characteristic Curves
pacman::p_load(pROC)
rocCurve <- roc(response = simulatedTest$class,
                predictor = simulatedTest$RFprob,
                ## This function assumes that the second 
                ## class is the event of interest, so we
                ## reverse the labels.
                levels = rev(levels(simulatedTest$class)))
auc(rocCurve)
ci(rocCurve)

plot(rocCurve, legacy.axes = TRUE)

# Lift Charts
labs <- c(RFprob = "Random Forest",
          QDAprob = "Quadratic Discriminant Analysis")
liftCurve <- lift(class ~ RFprob + QDAprob, data = simulatedTest, 
                  labels = labs)
liftCurve
## Add lattice options to produce a legend on top
pacman::p_load(lattice)
xyplot(liftCurve, 
       auto.key = list(columns = 2,
                       lines = TRUE,
                       points = FALSE))

# Calibrating Probabilities
calCurve <- calibration(class ~ RFprob + QDAprob, data = simulatedTest)
calCurve

xyplot(calCurve, auto.key = list(columns = 2))

## The glm() function models the probability of the second factor
## level, so the function relevel() is used to temporarily reverse the 
## factors levels.
sigmoidalCal <- glm(relevel(class, ref = "Class2") ~ QDAprob,
                    data = simulatedTrain,
                    family = binomial)
coef(summary(sigmoidalCal))
sigmoidProbs <- predict(sigmoidalCal,
                        newdata = simulatedTest[, "QDAprob", drop = FALSE],
                        type = "response")
simulatedTest$QDAsigmoid <- sigmoidProbs
# The Bayesian approach for calibration is to treat the training set class probabilities 
# to estimate the probabilities Pr[X] and Pr[X|Y = C_l]. 
pacman::p_load(klaR)
BayesCal <- NaiveBayes(class ~ QDAprob, data = simulatedTrain,
                       usekernel = TRUE)
## Like qda(), the predict function for this model creates
## both the classes and the probabilities
BayesProbs <- predict(BayesCal,
                      newdata = simulatedTest[, "QDAprob", drop = FALSE])
simulatedTest$QDABayes <- BayesProbs$posterior[, "Class1"]
## The probability values before and after calibration
head(simulatedTest[, c(5:6, 8, 9)])
# The option usekernel = TRUE allows a flexible function to model the probability distribution
# of the class probabilities.
# These new probabilities are evaluated using another plot:
calCurve2 <- calibration(class ~ QDAprob + QDABayes + QDAsigmoid,
                         data = simulatedTest)
xyplot(calCurve2)
