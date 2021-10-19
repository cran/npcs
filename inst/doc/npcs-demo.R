## ---- echo = FALSE------------------------------------------------------------
library(formatR)

## ---- eval=FALSE--------------------------------------------------------------
#  install.packages("npcs", repos = "http://cran.us.r-project.org")

## -----------------------------------------------------------------------------
library(npcs)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70)------------------------------
set.seed(123, kind = "L'Ecuyer-CMRG")
train.set <- generate_data(n = 1000, model.no = 1)
x <- train.set$x
y <- train.set$y

test.set <- generate_data(n = 2000, model.no = 1)
x.test <- test.set$x
y.test <- test.set$y

alpha <- c(0.05, NA, 0.01)
w <- c(0, 1, 0)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70)------------------------------
library(nnet)
fit.vanilla <- multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
error_rate(y.pred.vanilla, y.test)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70)------------------------------
fit.npmc.CX.logistic <- try(npcs(x, y, algorithm = "CX", classifier = "logistic", w = w, alpha = alpha))
fit.npmc.ER.logistic <- try(npcs(x, y, algorithm = "ER", classifier = "logistic", w = w, alpha = alpha, refit = TRUE))

# test error of NPMC-CX-logistic
y.pred.CX.logistic <- predict(fit.npmc.CX.logistic, x.test)
error_rate(y.pred.CX.logistic, y.test)

# test error of NPMC-ER-logistic
y.pred.ER.logistic <- predict(fit.npmc.ER.logistic, x.test)
error_rate(y.pred.ER.logistic, y.test)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70)------------------------------
fit.npmc.CX.lda <- try(npcs(x, y, algorithm = "CX", classifier = "lda", w = w, alpha = alpha))
fit.npmc.ER.lda <- try(npcs(x, y, algorithm = "ER", classifier = "lda", w = w, alpha = alpha, refit = TRUE))

fit.npmc.CX.rf <- try(npcs(x, y, algorithm = "CX", classifier = "randomforest", w = w, alpha = alpha))
fit.npmc.ER.rf <- try(npcs(x, y, algorithm = "ER", classifier = "randomforest", w = w, alpha = alpha, refit = TRUE))

# test error of NPMC-CX-LDA
y.pred.CX.lda <- predict(fit.npmc.CX.lda, x.test)
error_rate(y.pred.CX.lda, y.test)

# test error of NPMC-ER-LDA
y.pred.ER.lda <- predict(fit.npmc.ER.lda, x.test)
error_rate(y.pred.ER.lda, y.test)

# test error of NPMC-CX-RF
y.pred.CX.rf <- predict(fit.npmc.CX.rf, x.test)
error_rate(y.pred.CX.rf, y.test)

# test error of NPMC-ER-RF
y.pred.ER.rf <- predict(fit.npmc.ER.rf, x.test)
error_rate(y.pred.ER.rf, y.test)

