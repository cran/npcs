#' Gamma-synthetic minority over-sampling technique (gamma-SMOTE).
#'
#' gamma-SMOTE with some gamma in [0,1], which is a variant of the original SMOTE proposed by Chawla, N. V. et. al (2002). This can be combined with the NPMC methods proposed in Tian, Y., & Feng, Y. (2021). See Section 5.2.3 in Tian, Y., & Feng, Y. (2021) for more details.
#' @export
#' @param x the predictor matrix, where each row and column represents an observation and predictor, respectively.
#' @param y the response vector. Must be integers from 1 to K for some K >= 2. Can either be a numerical or factor vector.
#' @param dup_rate duplicate rate of original data. Default = 1, which finally leads to a new data set with twice sample size.
#' @param gamma the upper bound of uniform distribution used when generating synthetic data points in SMOTE. Can be any number between 0 and 1. Default = 0.5. When it equals to 1, gamma-SMOTE is equivalent to the original SMOTE (Chawla, N. V. et. al (2002)).
#' @param k the number of nearest neighbors during sampling process in SMOTE. Default = 5.
#' @return A list consisting of merged original and synthetic data, with two components x and y. x is the predictor matrix and y is the label vector.
#' @seealso \code{\link{npcs}}, \code{\link{predict.npcs}}, \code{\link{error_rate}}, and \code{\link{generate_data}}.
#' @references
#' Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357.
#'
#' Tian, Y., & Feng, Y. (2021). Neyman-Pearson Multi-class Classification via Cost-sensitive Learning. Submitted. Available soon on arXiv.
#'
#' @examples
#' \dontrun{
#' set.seed(123, kind = "L'Ecuyer-CMRG")
#' train.set <- generate_data(n = 200, model.no = 1)
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
#'
#' ## try NPMC-CX, NPMC-ER based on multinomial logistic regression, and vanilla multinomial
#' ## logistic regression without SMOTE. NPMC-ER outputs the infeasibility error information.
#' fit.npmc.CX <- try(npcs(x, y, algorithm = "CX", classifier = "logistic", w = w, alpha = alpha))
#' fit.npmc.ER <- try(npcs(x, y, algorithm = "ER", classifier = "logistic", w = w, alpha = alpha,
#' refit = TRUE))
#' fit.vanilla <- nnet::multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
#'
#' # test error of NPMC-CX based on multinomial logistic regression without SMOTE
#' y.pred.CX <- predict(fit.npmc.CX, x.test)
#' error_rate(y.pred.CX, y.test)
#'
#' # test error of vanilla multinomial logistic regression without SMOTE
#' y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
#' error_rate(y.pred.vanilla, y.test)
#'
#'
#' ## create synthetic data by 0.5-SMOTE
#' D.syn <- gamma_smote(x, y, dup_rate = 1, gamma = 0.5, k = 5)
#' x <- D.syn$x
#' y <- D.syn$y
#'
#' ## try NPMC-CX, NPMC-ER based on multinomial logistic regression, and vanilla multinomial logistic
#' ## regression with SMOTE. NPMC-ER can successfully find a solution after SMOTE.
#' fit.npmc.CX <- try(npcs(x, y, algorithm = "CX", classifier = "logistic", w = w, alpha = alpha))
#' fit.npmc.ER <- try(npcs(x, y, algorithm = "ER", classifier = "logistic", w = w, alpha = alpha,
#' refit = TRUE))
#' fit.vanilla <- nnet::multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
#'
#' # test error of NPMC-CX based on multinomial logistic regression with SMOTE
#' y.pred.CX <- predict(fit.npmc.CX, x.test)
#' error_rate(y.pred.CX, y.test)
#'
#' # test error of NPMC-ER based on multinomial logistic regression with SMOTE
#' y.pred.ER <- predict(fit.npmc.ER, x.test)
#' error_rate(y.pred.ER, y.test)
#'
#' # test error of vanilla multinomial logistic regression wit SMOTE
#' y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
#' error_rate(y.pred.vanilla, y.test)
#' }

gamma_smote <- function(x, y, dup_rate = 1, gamma = 0.5, k = 5) {
  if (dup_rate > k) {
    replace.sample <- TRUE
  } else {
    replace.sample <- FALSE
  }
  i <- NULL
  indicator.fac <- apply(x, 2, function(v){is.factor(v)})
  nb.index <- knearest(D = x, P = x, n_clust = k)
  x.syn <- foreach(i = 1:nrow(x), .combine = "rbind") %do% {
    ind <- sample(nb.index, size = dup_rate, replace = replace.sample)
    u <- runif(n = dup_rate, min = 0, max = gamma)
    M <- (1-u)*matrix(rep(x[i, , drop = F], each = dup_rate), nrow = dup_rate) + u*x[ind, , drop = F]
    M[, indicator.fac] <- matrix(as.numeric(M[, indicator.fac] > 0.5), nrow = dup_rate)
    M
  }
  y.syn <- rep(y, each = dup_rate)

  x.syn <- rbind(x, x.syn)
  y.syn <- factor(c(y, y.syn))


  return(list(x = x.syn, y = y.syn))
}
