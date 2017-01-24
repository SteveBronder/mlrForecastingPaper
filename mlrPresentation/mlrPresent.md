Forecasting in the MLR Framework
========================================================
author: Steve Bronder
date: October 11th, 2016
autosize: true

What is Forecasting?
========================================================

- Predictions of future based on past trends
- What happens Vs. What happens tomorrow
- Stocks, earthquakes, neuroscience

Goal: Make Forecasting Simple
========================================================

> "We need to stop teaching abstinence and start teaching safe statistics"
- Hadley Wickham

Ex: Demeaning the whole data set before CV

- Problem: Forecasting can be dangerous
- Insight: Need a framework for 'safe forecasting'
- Solution: Use ML Framework in forecasting

The Modeling Process
========================================================

<img src="flownames.jpeg" alt="Drawing" style="width: 90%; height: 90%"/>

***

- MLR automates this pipeline
- Make forecasting safer by using the pipeline

Example Data
======================================================


```r
library(Quandl)
library(xts)
aapl <- Quandl("YAHOO/AAPL", api_key="UG7wmFCm6zMyq1xhW9Re")
aaplXts <- xts(aapl$Close, order.by = as.POSIXlt(aapl$Date))
colnames(aaplXts) <- "Close"
aaplXtsTrain <- aaplXts[1:9000,]
aaplXtsTest  <- aaplXts[9001:9035,]
```

Plot of aapl Stock
=====================================================



<img src="training_aapl.png" alt="Drawing" style="width: 90%; height: 90%"/>

***

<img src="testing_aapl.png" alt="Drawing" style="width: 90%; height: 90%"/>

Creating a Forecasting Task
========================================================

- Task: Keeps data and meta-data for ML task


```r
library(mlr)
aaplTask <- makeForecastRegrTask(
  id = "Forecast aapl Closing Price",
  data = aaplXtsTrain,
  target  = "Close",
  frequency = 7L)
```

Creating a Forecasting Task: Info
========================================================

```r
aaplTask
```

```
Task: Forecast aapl Closing Price
Type: fcregr
Target: Close
Observations: 9000
Dates:
 Start: 1980-12-12 
 End:   2016-08-19
Frequency: 7
Features:
numerics  factors  ordered 
       0        0        0 
Missings: FALSE
Has weights: FALSE
Has blocking: FALSE
```


Making a Forecasting Learner
======================================================


```r
garch.mod =makeLearner("fcregr.garch", 
                      model = "sGARCH",
                      garchOrder = c(5,5),
                      distribution.model = "sged",
                      armaOrder = c(6,6),
                      n.ahead = 35,
                      predict.type = "quantile")
```

Making a Forecasting Learner
======================================================


```r
garch.mod 
```

```
Learner fcregr.garch from package rugarch
Type: fcregr
Name: Generalized AutoRegressive Conditional Heteroskedasticity; Short name: garch
Class: fcregr.garch
Properties: numerics,quantile
Predict-Type: quantile
Hyperparameters: model=sGARCH,garchOrder=5,5,distribution.model=sged,armaOrder=6,6,n.ahead=35
```

Train a Forecast Learner
======================================================


```r
garch.train <- train(garch.mod, aaplTask)
garch.train
```

Predict With a Forecast Learner
======================================================

```r
predAapl <- predict(garch.train, newdata = as.data.frame(aaplXtsTest))
performance(predAapl, measures = mase, task = aaplTask)
```

```
      mase 
0.00701081 
```

Prediction Plot
=====================================================

<img src="mlrPresent-figure/predplotArima-1.png" title="plot of chunk predplotArima" alt="plot of chunk predplotArima" style="display: block; margin: auto;" />



Tuning a Model
=====================================================


```r
# Make a tuning grid for GARCH
par_set = makeParamSet(
  makeDiscreteParam(id = "model",
                    values = c("sGARCH", "csGARCH", "fGARCH")),
  makeDiscreteParam("submodel", values = c("GARCH","TGARCH","AVGARCH"),requires = quote(model == 'fGARCH') ),
  makeIntegerVectorParam(id = "garchOrder", len = 2L,
                         lower = 1, upper = 8),
  makeIntegerVectorParam(id = "armaOrder", len = 2L,
                         lower = 1, upper = 9),
  makeLogicalParam(id = "include.mean"),
  makeLogicalParam(id = "archm"),
  makeDiscreteParam(id = "distribution.model",
                    values = c("norm","std","jsu", "sged")),
  makeDiscreteParam(id = "stationarity", c(0,1)),
  makeDiscreteParam(id = "fixed.se", c(0,1))
)
```

Making a Resample Scheme
========================================


```r
resampDesc = makeResampleDesc("GrowingCV", horizon = 35L,
                              initial.window = .9,
                              size = nrow(getTaskData(aaplTask)),
                              skip = .01)
resampDesc
```

```
Window description:
 growing with 10 iterations:
 8100 observations in initial window and 35 horizon.
Predict: test
Stratification: FALSE
```

Example of Windowing Resample
======================================================
<center>
<img src="caret_window.png" alt="Drawing" style="width: 1200px; height: 600px"/>
</center>

Making a Tuning Control
===========================================


```r
ctrl <- makeTuneControlIrace(maxExperiments = 350)
```

Tuning Over Parameter Space
===========================================


```r
library("parallelMap")
parallelStart("multicore",3)
configureMlr(on.learner.error = "warn")
set.seed(1234)

garch.mod = makeLearner("fcregr.garch", n.ahead = 35, solver = 'hybrid')
garch.res = tuneParams(garch.mod, task = aaplTask,
                 resampling = resampDesc, par.set = par_set,
                 control = ctrl,
                 measures = mase)
parallelStop()

garch.final = setHyperPars(makeLearner("fcregr.garch", n.ahead = 35, solver = 'nloptr', solver.control = list(maxeval = 200000, solver = 10), predict.type = "quantile"),par.vals = garch.res$x)

garch.train = train(garch.final, aaplTask)
garch.pred = predict(garch.train, newdata = aaplXtsTest)
performance(garch.pred, measures = mase, task = aaplTask)
matplot(garch.pred$data, type = "l")
save(garch.res, file ="./garch_tune_mod.RData")
```


Using an ML Model
=========================================


```r
aaplRegTask <- makeRegrTask(
  id = "Forecast aapl Closing Price",
  data = as.data.frame(aaplXtsTrain,rownames = index(aaplXtsTrain)),
  target  = "Close")

aaplLagTask = createLagDiffFeatures(aaplRegTask, lag = 1L:665L, difference = 1L, na.pad = FALSE)


## Trying Support Vector Machines
xg_learner <- makeLearner("regr.xgboost")

xg_param_set <- makeParamSet(
  makeDiscreteParam(id = "booster", values = c("gbtree")),
  makeNumericLearnerParam(id = "eta", lower = 0, upper = 1),
  makeNumericLearnerParam(id = "gamma", lower = 0, upper = 100),
  makeIntegerLearnerParam(id = "max_depth",  lower = 100L, upper = 300),
  makeIntegerLearnerParam(id = "nrounds", lower = 100L, upper = 300)
)
# A pretty good model had something like
# [Tune-x] 1: booster=gbtree; eta=0.248; gamma=61.4; max_depth=425; subsample=0.456;
# colsample_bytree=0.971; colsample_bylevel=0.612; lambda=18.5; lambda_bias=9.49;
# alpha=5.62; nrounds=493

ctrl <- makeTuneControlIrace(maxExperiments = 200)

library("parallelMap")
parallelStart("multicore",3)
configureMlr(on.learner.error = "warn")
tune_mod <- tuneParams(learner = xg_learner, task = aaplLagTask,
                       measures = mase, resampling = resampDesc,
                       par.set = xg_param_set, control = ctrl )
parallelStop()

gbm_final = setHyperPars(xg_learner, par.vals = tune_mod$x)
gbm_train <- train(gbm_final, aaplLagTask)
gbm_fore = forecast(gbm_train, h = 35, newdata = aaplXtsTest)
performance(gbm_fore, mase, task = aaplLagTask)
save(tune_mod, file = "./gbm_tune_mod.RData")
```

Make Resampling Scheme
===========================================


```r
resampDesc = makeResampleDesc("GrowingCV", horizon = 35L,
                               initialWindow = 7000L,
                               size = nrow(getTaskData(aaplLagTask)), skip = 35L)
```

Make Tuning Set and Search Scheme
===========================================

```r
library(mlr)
ps = makeParamSet(
  makeDiscreteParam("distribution", values = c("gaussian","laplace", "tdist")),
  makeIntegerParam("n.trees", lower = 1, upper = 10, trafo = function(x) x * 1000),
  makeIntegerParam("interaction.depth", lower = 1, upper = 5, trafo = function(x) x * 5),
  makeNumericParam("shrinkage", lower = 1E-5, upper = 1E-2)
)
ctrl = makeTuneControlIrace(maxExperiments = 200L)
```
Tuning GBM Model
===========================================











```
Error in checkClass(x, classes, ordered) : object 'aaplLagTask' not found
```
