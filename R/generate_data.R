#' Generate the data.
#'
#' Generate the data from two simulation cases in Tian, Y., & Feng, Y. (2021).
#' @export
#' @param n the generated sample size. Default = 1000.
#' @param model.no the model number in Tian, Y., & Feng, Y. (2021). Can be 1 or 2. Default = 1.
#' @return A list with two components x and y. x is the predictor matrix and y is the label vector.
#' @seealso \code{\link{npcs}}, \code{\link{predict.npcs}}, \code{\link{error_rate}}, and \code{\link{gamma_smote}}.
#' @references
#' Tian, Y., & Feng, Y. (2021). Neyman-Pearson Multi-class Classification via Cost-sensitive Learning. Submitted. Available soon on arXiv.
#'
#' @examples
#' set.seed(123, kind = "L'Ecuyer-CMRG")
#' train.set <- generate_data(n = 1000, model.no = 1)
#' x <- train.set$x
#' y <- train.set$y
#'
generate_data <- function(n = 1e3, model.no = 1)
{
  if (model.no == 1) {
    p <- 5
    X <- matrix(rnorm(n*p), nrow = n, ncol = p)
    mu <- matrix(nrow = 3, ncol = p)
    mu[1, ] <- c(-1,2,1,1,1)
    mu[2, ] <- c(1,1,0,2,0)
    mu[3, ] <- c(2,-1,-1,0,0)
    Y <- sample(1:3, size = n, replace = TRUE, prob = c(0.3, 0.3, 0.4))
    X[Y == 1, ] <- X[Y == 1, ] + rep(mu[1, ], each = sum(Y == 1))
    X[Y == 2, ] <- X[Y == 2, ] + rep(mu[2, ], each = sum(Y == 2))
    X[Y == 3, ] <- X[Y == 3, ] + rep(mu[3, ], each = sum(Y == 3))
  } else if (model.no == 2) {
    p <- 5
    Sigma <- outer(1:p, 1:p, function(i,j){0.1^(I(i!=j))})
    R <- chol(Sigma)
    X <- matrix(rnorm(n*p), nrow = n, ncol = p) %*% R
    mu <- matrix(nrow = 4, ncol = p)

    mu[1, ] <- c(1,-2,0,-1,1)
    mu[2, ] <- c(-1,1,-2,-1,1)
    mu[3, ] <- c(2,0,-1,1,-1)
    mu[4, ] <- c(1,0,1,2,-2)

    Y <- sample(1:4, size = n, replace = TRUE, prob = c(0.1, 0.2, 0.3, 0.4))
    X[Y == 1, ] <- X[Y == 1, ] + rep(mu[1, ], each = sum(Y == 1))
    X[Y == 2, ] <- X[Y == 2, ] + rep(mu[2, ], each = sum(Y == 2))
    X[Y == 3, ] <- X[Y == 3, ] + rep(mu[3, ], each = sum(Y == 3))
    X[Y == 4, ] <- X[Y == 4, ] + rep(mu[4, ], each = sum(Y == 4))
  }

  Y <- factor(Y)


  list(x = X, y = Y)
}
