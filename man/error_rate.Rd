% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/error_rate.R
\name{error_rate}
\alias{error_rate}
\title{Calculate the error rates for each class.}
\usage{
error_rate(y.pred, y, class.names = NULL)
}
\arguments{
\item{y.pred}{the predicted labels.}

\item{y}{the true labels.}

\item{class.names}{the names of classes. Should be a string vector. Default = NULL, which will set the name as 1, ..., K, where K is the number of classes.}
}
\value{
A vector of the error rate for each class. The vector name is the same as \code{class.names}.
}
\description{
Calculate the error rate for each class given the predicted labels and true labels.
}
\examples{
# data generation
set.seed(123, kind = "L'Ecuyer-CMRG")
train.set <- generate_data(n = 1000, model.no = 1)
x <- train.set$x
y <- train.set$y

test.set <- generate_data(n = 1000, model.no = 1)
x.test <- test.set$x
y.test <- test.set$y

library(nnet)
fit.vanilla <- multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
error_rate(y.pred.vanilla, y.test)
}
\references{
Tian, Y., & Feng, Y. (2021). Neyman-Pearson Multi-class Classification via Cost-sensitive Learning. Submitted. Available soon on arXiv.
}
\seealso{
\code{\link{npcs}}, \code{\link{predict.npcs}}, \code{\link{generate_data}}, \code{\link{gamma_smote}}.
}
