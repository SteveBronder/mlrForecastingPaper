library(mlr)
climate.task = makeForecastRegrTask(id = "M4 Climate Data",
                                    data = m4.train,
                                    target = "target_var",
                                    frequency = 7L)
climate.task

parSet = makeParamSet(
  makeDiscreteParam("model", values = c("ANN", "MNN", "ZNN",
                                        "AAN", "MAN", "ZAN",
                                        "AMN", "MMN", "ZMN",
                                        "AZN", "MZN", "ZZN",
                                        "ANA", "MNA", "ZNA",
                                        "AAA", "MAA", "ZAA",
                                        "AMA", "MMA", "ZMA",
                                        "AZA")),
  makeLogicalParam("damped"),
  makeLogicalParam("additive.only"),
  makeLogicalParam("biasadj"),
  makeDiscreteParam("opt.crit", values = c("mase", "amse", "sigma", "mae", "lik")),
  makeDiscreteParam("ic", values = c("aicc", "aic", "bic")),
  makeLogicalParam("allow.multiplicative.trend")
)

resampDesc = makeResampleDesc("GrowingCV", horizon = 35L,
                              initial.window = .7,
                              size = nrow(getTaskData(climate.task)),
                              skip = .025)
resampDesc

ctrl = makeTuneControlIrace(maxExperiments = 1500L)

library("parallelMap")
parallelStart("multicore", 8)
configureMlr(on.learner.error = "warn")
set.seed(1234)
etsTune = tuneParams(makeLearner("fcregr.ets", h = 35),
                       task = climate.task,
                       resampling = resampDesc, par.set = parSet,
                       control = ctrl, measures = mase)
parallelStop()
# Output the test mean of best model
etsTune$y

ets.lrn = setHyperPars(makeLearner("fcregr.ets",
                                     h = 35),
                         par.vals = etsTune$x)
ets.final = train(ets.lrn, climate.task)

climate.pred = predict(ets.final, newdata = m4.test)
performance(climate.pred, measures = mase, task = climate.task)


forecast.train.ets = forecast::ets(ts(m4.train,frequency = 7))
forecast.ets = forecast::forecast(forecast.train.ets,h=35)
# Calculate MASE
forecast::accuracy(forecast.ets, x= ts(m4.test, end = c(93,2), frequency = 7))[2,6,drop=FALSE]


lrn = makeLearner("mfcregr.BigVAR", p = 9, struct = "SparseLag", gran = c(35, 50), recursive = TRUE, h = 35, n.ahead = 35)
trn.lrn = train(lrn, multfore.task)

pred.mult <- predict(trn.lrn, newdata = eu.test)
performance(pred.mult, mase, task = multfore.task)
