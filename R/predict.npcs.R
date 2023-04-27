#' Predict new labels from new data based on the fitted NPMC classifier.
#'
#' Predict new labels from new data based on the fitted NPMC classifier, which belongs to S3 class "npcs".
#' @export
#' @param object the model object for prediction 
#' @param newx input feature data
#' @param ... arguments to pass down
predict.npcs <- function(object, newx, ...) {
  if (is.matrix(newx)) newx <- data.frame(newx)  
  posterior.test <- predict(object$fit, newdata = newx, type = "prob")
  object$lambda[is.na(object$lambda)] <- 0
  ck <- (object$lambda + object$w)/object$pik
  cost_posterior.test <- t(t(posterior.test)*ck)
  pred.test <- as.numeric(apply(cost_posterior.test, 1, which.max))
  return(pred.test)
}
