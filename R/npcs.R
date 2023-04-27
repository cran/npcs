#' Fit a multi-class Neyman-Pearson classifier with error controls via cost-sensitive learning.
#'
#' Fit a multi-class Neyman-Pearson classifier with error controls via cost-sensitive learning. This function implements two algorithms proposed in Tian, Y. & Feng, Y. (2021). The problem is minimize a linear combination of P(hat(Y)(X) != k| Y=k) for some classes k while controlling P(hat(Y)(X) != k| Y=k) for some classes k. See Tian, Y. & Feng, Y. (2021) for more details.
#' @export
#' @importFrom stats predict
#' @importFrom stats runif
#' @importFrom stats optimize
#' @importFrom stats rnorm
#' @importFrom dfoptim nmkb
#' @importFrom dfoptim hjkb
#' @importFrom magrittr %>%
#' @importFrom caret train
#' @importFrom caret trainControl
#' @importFrom smotefamily knearest
#' @importFrom foreach foreach
#' @importFrom foreach %do%
#' @importFrom formatR tidy_eval
#' @importFrom nnet multinom
#' @param x the predictor matrix of training data, where each row and column represents an observation and predictor, respectively.
#' @param y the response vector of training data. Must be integers from 1 to K for some K >= 2. Can be either a numerical or factor vector.
#' @param algorithm the NPMC algorithm to use. String only. Can be either "CX" or "ER", which implements NPMC-CX or NPMC-ER in Tian, Y. & Feng, Y. (2021).
#' @param classifier which model to use for estimating the posterior distribution P(Y|X = x). String only.
#' @param seed random seed
#' @param w the weights in objective function. Should be a vector of length K, where K is the number of classes.
#' @param alpha the levels we want to control for error rates of each class. Should be a vector of length K, where K is the number of classes. Use NA if if no error control is imposed for specific classes.
#' @param trControl list; resampling method
#' @param tuneGrid list; for hyperparameters tuning or setting
#' @param split.ratio the proportion of data to be used in searching lambda (cost parameters). Should be between 0 and 1. Default = 0.5. Only useful when \code{algorithm} = "ER".
#' @param split.mode two different modes to split the data for NPMC-ER. String only. Can be either "per-class" or "merged". Default = "per-class". Only useful when \code{algorithm} = "ER".
#' \itemize{
#' \item per-class: split the data by class.
#' \item merged: split the data as a whole.
#' }
#' @param tol the convergence tolerance. Default = 1e-06. Used in the lambda-searching step. The optimization is terminated when the step length of the main loop becomes smaller than \code{tol}. See pages of \code{\link[dfoptim]{hjkb}} and \code{\link[dfoptim]{nmkb}} for more details.
#' @param refit whether to refit the classifier using all data after finding lambda or not. Boolean value. Default = TRUE. Only useful when \code{algorithm} = "ER".
#' @param protect whether to threshold the close-zero lambda or not. Boolean value. Default = TRUE. This parameter is set to avoid extreme cases that some lambdas are set equal to zero due to computation accuracy limit. When \code{protect} = TRUE, all lambdas smaller than 1e-03 will be set equal to 1e-03.
#' @param opt.alg optimization method to use when searching lambdas. String only. Can be either "Hooke-Jeeves" or "Nelder-Mead". Default = "Hooke-Jeeves".
#' @return An object with S3 class \code{"npcs"}.
#' \item{lambda}{the estimated lambda vector, which consists of Lagrangian multipliers. It is related to the cost. See Section 2 of Tian, Y. & Feng, Y. (2021) for details.}
#' \item{fit}{the fitted classifier.}
#' \item{classifier}{which classifier to use for estimating the posterior distribution P(Y|X = x).}
#' \item{algorithm}{the NPMC algorithm to use.}
#' \item{alpha}{the levels we want to control for error rates of each class.}
#' \item{w}{the weights in objective function.}
#' \item{pik}{the estimated marginal probability for each class.}
#' @seealso \code{\link{predict.npcs}}, \code{\link{error_rate}}, \code{\link{generate_data}}, \code{\link{gamma_smote}}.
#' @references
#' Tian, Y., & Feng, Y. (2021). Neyman-Pearson Multi-class Classification via Cost-sensitive Learning. Submitted. Available soon on arXiv.
#'
#' @examples
#' # data generation: case 1 in Tian, Y., & Feng, Y. (2021) with n = 1000
#' set.seed(123, kind = "L'Ecuyer-CMRG")
#' train.set <- generate_data(n = 1000, model.no = 1)
#' x <- train.set$x
#' y <- train.set$y
#'
#' test.set <- generate_data(n = 1000, model.no = 1)
#' x.test <- test.set$x
#' y.test <- test.set$y
#'
#' # contruct the multi-class NP problem: case 1 in Tian, Y., & Feng, Y. (2021)
#' alpha <- c(0.05, NA, 0.01)
#' w <- c(0, 1, 0)
#' \donttest{
#' # try NPMC-CX, NPMC-ER, and vanilla multinomial logistic regression
#' fit.vanilla <- nnet::multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
#' fit.npmc.CX <- try(npcs(x, y, algorithm = "CX", classifier = "multinom", 
#' w = w, alpha = alpha))
#' fit.npmc.ER <- try(npcs(x, y, algorithm = "ER", classifier = "multinom", 
#' w = w, alpha = alpha, refit = TRUE))
#' # test error of vanilla multinomial logistic regression
#' y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
#' error_rate(y.pred.vanilla, y.test)
#' # test error of NPMC-CX
#' y.pred.CX <- predict(fit.npmc.CX, x.test)
#' error_rate(y.pred.CX, y.test)
#' # test error of NPMC-ER
#' y.pred.ER <- predict(fit.npmc.ER, x.test)
#' error_rate(y.pred.ER, y.test)
#' }
npcs <- function(x, y, algorithm = c("CX", "ER"), classifier,seed=1,
                 w, alpha, trControl=list(), tuneGrid=list(),split.ratio = 0.5, 
                 split.mode = c("by-class", "merged"), tol = 1e-6, refit = TRUE, 
                 protect = TRUE, opt.alg = c("Hooke-Jeeves", "Nelder-Mead")) {
  algorithm <- match.arg(algorithm)
  # classifier <- match.arg(classifier)
  split.mode <- match.arg(split.mode)
  opt.alg <- match.arg(opt.alg)
  n <- length(y)
  K <- length(unique(y))
  w.ori <- w
  w <- w/sum(w)
  index <- which(!is.na(alpha))
  stopifnot(class(trControl)=="list", class(tuneGrid)=="list", !("y" %in% colnames(x)))      # feature name cannot be y because formula hard coding
  if (algorithm == "CX") {
    pik <- as.numeric(table(y)/n)
    df <- data.frame(x, y=y)
    fit <- modeling(data = df,classifier=classifier, trControl=trControl, tuneGrid=tuneGrid,seed=seed)  
    posterior <- predict(fit, newdata =df, type = "prob")   
    if (length(index) == 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- optimize(f = obj.CX, lower = 0, maximum = T, upper = 200, alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, tol = tol)
        # lambda <- nmkb1(par = rep(0.0001, length(index)), fn = obj.CX, upper = rep(200, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb1(par = rep(0, length(index)), fn = obj.CX, upper = rep(200, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      }
    } else if (length(index) > 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- nmkb(par = rep(0.0001, length(index)), fn = obj.CX, upper = rep(200, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb(par = rep(0, length(index)), fn = obj.CX, upper = rep(200, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      }
    }
    
  }
  
  if (algorithm == "ER") {
    if (split.mode == "merged") {
      ind <- sample(n, floor(n*split.ratio)) # the indices of samples used to estimate lambda
      pik <- as.numeric(table(y[-ind])/length(y[-ind]))
    } else {
      ind <- Reduce("c", sapply(1:K, function(k){
        ind.k <- which(y == k)
        sample(ind.k, floor(length(ind.k)*split.ratio))
      }, simplify = F))
      pik <- as.numeric(table(y[-ind])/length(y[-ind]))
    }
    fit <- modeling(data=data.frame(x[-ind,], y=y[-ind]),classifier=classifier, 
                    trControl=trControl, tuneGrid=tuneGrid,seed=seed) 
    posterior <- predict(fit, newdata = data.frame(x[ind,], y=y[ind]), type = "prob")
    if (refit) {
      df <- data.frame(x, y=y) 
      fit <- modeling(data = df,classifier=classifier, trControl=trControl, tuneGrid=tuneGrid, seed=seed)
      pik <- as.numeric(table(y)/length(y))
    }
    if (length(index) == 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- optimize(f = obj.ER, lower = 0, maximum = T, upper = 200, alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind], tol = tol)
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb1(par = rep(0, length(index)), fn = obj.ER, upper = rep(200, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
      }
    } else if (length(index) > 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- nmkb(par = rep(0.0001, length(index)), fn = obj.ER, upper = rep(200, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb(par = rep(0, length(index)), fn = obj.ER, upper = rep(200, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
      }
    }
  }
  
  if (length(index) == 1 && opt.alg == "Nelder-Mead") {
    obj.value <- lambda$objective
    lambda <- lambda$maximum
  } else {
    obj.value <- lambda$value
    lambda <- lambda$par
  }
  
  if (obj.value > 1) {
    stop("The NP problem is infeasible!")
  }
  
  if (protect) {
    lambda[lambda <= 1e-3] <- 1e-3
  }
  
  lambda.full <- rep(NA, length(alpha))
  lambda.full[index] <- lambda
  
  L <- list(lambda = lambda.full*sum(w.ori), fit = fit, classifier = classifier, algorithm = algorithm, alpha = alpha, w = w.ori, pik = pik)
  class(L) <- "npcs"
  
  
  return(L)
}
