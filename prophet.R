library(prophet)
m4_train_prophet <- data.frame(y = coredata(m4.train), ds = index(m4.train))
colnames(m4_train_prophet) <- c("y","ds")
m <- prophet(m4_train_prophet)
future <- make_future_dataframe(m, periods = 36, include_history =FALSE)
tail(future)
pred_prophet <- predict(m, future)

pred_prophet <- data.frame(prophet_response = pred_prophet$yhat,
                           truth = coredata(m4.test))


