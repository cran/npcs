## ---- echo = FALSE------------------------------------------------------------
library(formatR)

## ---- eval=FALSE--------------------------------------------------------------
#  # install.packages("npcs_0.1.1.tar.gz", repos=NULL, type='source')
#  # install.packages("npcs", repos = "http://cran.us.r-project.org")

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

## ----message=FALSE, list(width.cutoff=70), tidy=TRUE, tidy.opts=list(width.cutoff=70)----
fit.npmc.CX.logistic <- try(npcs(x, y, algorithm = "CX", classifier = "multinom", w = w, alpha = alpha))
fit.npmc.ER.logistic <- try(npcs(x, y, algorithm = "ER", classifier = "multinom", w = w, alpha = alpha, refit = TRUE))

# test error of NPMC-CX-logistic
y.pred.CX.logistic <- predict(fit.npmc.CX.logistic, x.test)
error_rate(y.pred.CX.logistic, y.test)

# test error of NPMC-ER-logistic
y.pred.ER.logistic <- predict(fit.npmc.ER.logistic, x.test)
error_rate(y.pred.ER.logistic, y.test)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70)------------------------------
fit.npmc.CX.lda <- try(npcs(x, y, algorithm = "CX", classifier = "lda", w = w, alpha = alpha))
fit.npmc.ER.lda <- try(npcs(x, y, algorithm = "ER", classifier = "lda", w = w, alpha = alpha, refit = TRUE))
library(gbm)
fit.npmc.CX.gbm <- try(npcs(x, y, algorithm = "CX", classifier = "gbm", w = w, alpha = alpha))
fit.npmc.ER.gbm <- try(npcs(x, y, algorithm = "ER", classifier = "gbm", w = w, alpha = alpha, refit = TRUE))

# test error of NPMC-CX-LDA
y.pred.CX.lda <- predict(fit.npmc.CX.lda, x.test)
error_rate(y.pred.CX.lda, y.test)

# test error of NPMC-ER-LDA
y.pred.ER.lda <- predict(fit.npmc.ER.lda, x.test)
error_rate(y.pred.ER.lda, y.test)

# test error of NPMC-CX-GBM
y.pred.CX.gbm <- predict(fit.npmc.CX.gbm, x.test)
error_rate(y.pred.CX.gbm, y.test)

# test error of NPMC-ER-GBM
y.pred.ER.gbm <- predict(fit.npmc.ER.gbm, x.test)
error_rate(y.pred.ER.gbm, y.test)

## -----------------------------------------------------------------------------
# 5-fold cross validation with tuning parameters k = 5,7,9
fit.npmc.CX.knn <- npcs(x, y, algorithm = "CX", classifier = "knn", w = w, 
                            alpha = alpha,seed = 1, 
                            trControl = list(method="cv", number=3), 
                            tuneGrid = list(k=c(5,7,9)))
# the optimal hypterparameter is k=9
fit.npmc.CX.knn$fit
y.pred.CX.knn <- predict(fit.npmc.CX.knn, x.test)
error_rate(y.pred.CX.knn, y.test)

## ----warning=FALSE------------------------------------------------------------
cv.npcs.knn <- cv.npcs(x, y, classifier = "knn", w = w, alpha = alpha,
                         # fold=5, stratified=TRUE, partition_ratio = 0.7
                         # resample=c("bootstrapping", "cv"),seed = 1, 
                         # plotit=TRUE, trControl=list(), tuneGrid=list(), 
                         # verbose=TRUE
                         )
cv.npcs.knn$summaries
cv.npcs.knn$plot

