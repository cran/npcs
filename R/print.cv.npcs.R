#' Print the cv.npcs object.
#' @export
#' @method print cv.npcs
#' @param x fitted cv.npcs object using \code{cv.npcs}.
#' @param ... additional arguments.


print.cv.npcs <- function(x, ...) {
  x <- x[c("summaries", "plot")]
  class(x) <- "list"
  # attr(x, "class") = "list"
  print(x)
}
