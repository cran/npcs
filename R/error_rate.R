#' Calculate the error rates for each class.
#'
#' Calculate the error rate for each class given the predicted labels and true labels.
#' @export
#' @param y.pred the predicted labels.
#' @param y the true labels.
#' @param class.names the names of classes. Should be a string vector. Default = NULL, which will set the name as 1, ..., K, where K is the number of classes.
#' @return A vector of the error rate for each class. The vector name is the same as \code{class.names}.
#' @seealso \code{\link{npcs}}, \code{\link{predict.npcs}}, \code{\link{generate_data}}, \code{\link{gamma_smote}}.
#' @references
#' Tian, Y., & Feng, Y. (2021). Neyman-Pearson Multi-class Classification via Cost-sensitive Learning. Submitted. Available soon on arXiv.
#'
#' @examples
#' # data generation
#' set.seed(123, kind = "L'Ecuyer-CMRG")
#' train.set <- generate_data(n = 1000, model.no = 1)
#' x <- train.set$x
#' y <- train.set$y
#'
#' test.set <- generate_data(n = 1000, model.no = 1)
#' x.test <- test.set$x
#' y.test <- test.set$y
#'
#' library(nnet)
#' fit.vanilla <- multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
#' y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
#' error_rate(y.pred.vanilla, y.test)

error_rate <- function(y.pred, y, class.names = NULL) {
  if (is.null(class.names)) {
    class.names <- as.numeric(levels(as.factor(y)))
  }


  error_rate <- sapply(1:length(class.names),function(i){
    length(y.pred[y == class.names[i] & y.pred != class.names[i]])/length(y[y == class.names[i]])
  })

  names(error_rate) <- class.names

  return(error_rate)
}
