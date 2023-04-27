#' Compare the performance of the NPMC-CX, NPMC-ER, and vanilla models through cross-validation or bootstrapping methods
#' 
#' Compare the performance of the NPMC-CX, NPMC-ER, and vanilla models through cross-validation or bootstrapping methods. The function will return a summary of evaluation which includes various evaluation metrics, and visualize the class-specific error rates.
#' @export
#' @importFrom dplyr dense_rank
#' @importFrom dplyr all_of
#' @importFrom foreach foreach
#' @importFrom foreach %do%
#' @importFrom dplyr dense_rank
#' @importFrom dplyr mutate
#' @importFrom dplyr group_by
#' @importFrom dplyr ungroup
#' @importFrom dplyr vars
#' @importFrom dplyr summarize_at
#' @importFrom magrittr %>%
#' @importFrom stats predict
#' @importFrom stats aggregate
#' @importFrom stats as.formula
#' @importFrom tidyr pivot_longer
#' @importFrom forcats fct_relevel 
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 position_dodge
#' @importFrom ggplot2 geom_boxplot
#' @importFrom ggplot2 geom_hline
#' @importFrom ggplot2 ggtitle
#' @importFrom ggplot2 geom_violin
#' @param x matrix; the predictor matrix of complete data
#' @param y numeric/factor/string; the response vector of complete data.
#' @param classifier string; Model to use for npcs function
#' @param alpha the levels we want to control for error rates of each class. The length must be equal to the number of classes
#' @param w the weights in objective function. Should be a vector of length K, where K is the number of classes.
#' @param fold integer; number of folds in CV or number of bootstrapping iterations, default=5
#' @param stratified logical; if TRUE, sample will be split into groups based on the proportion of response vector
#' @param partition_ratio numeric; the proportion of data to be used for model construction when parameter resample=="bootstrapping"
#' @param resample string; the resampling method
#' \itemize{
#' \item bootstrapping: bootstrapping, which iteration number is set by parameter "fold"
#' \item cv: cross validation, the number of folds is set by parameter "fold"
#' }
#' @param seed random seed 
#' @param verbose logical; if TRUE, cv.npcs will print the progress. If FALSE, the model will remain silent
#' @param plotit logical; if TRUE, the output list will return a box plot summarizing the error rates of vanilla model and NPMC model
#' @param trControl list; resampling method within each fold
#' @param tuneGrid list; for hyperparameters tuning or setting
#' @examples
#' # data generation: case 1 in Tian, Y., & Feng, Y. (2021) with n = 1000
#' set.seed(123, kind = "L'Ecuyer-CMRG")
#' train.set <- generate_data(n = 1000, model.no = 1)
#' x <- train.set$x
#' y <- train.set$y
#' test.set <- generate_data(n = 2000, model.no = 1)
#' x.test <- test.set$x
#' y.test <- test.set$y
#' alpha <- c(0.05, NA, 0.01)
#' w <- c(0, 1, 0)
#' # contruct the multi-class NP problem
#' \donttest{
#' cv.npcs.knn <- cv.npcs(x, y, classifier = "knn", w = w, alpha = alpha)
#' # result summary and visualization
#' cv.npcs.knn$summaries
#' cv.npcs.knn$plot
#' }
cv.npcs <- function(x, y, classifier, alpha, w, fold=5, stratified=TRUE, 
                      partition_ratio = 0.7, resample=c("bootstrapping", "cv"),seed = 1,
                      verbose=TRUE, plotit=TRUE, trControl=list(),
                      tuneGrid=list()){
  resample <- match.arg(resample)
  stopifnot(length(partition_ratio)==1, partition_ratio<1 & partition_ratio>0, 
            length(fold)==1, (fold==round(fold)&fold>0)
            )
  k <- length(unique(y))          # number of classes
  n <- length(y)                  # number of observations
  idxs <- vector(mode = "list")         # bootstrapping indices
  y_label <- if (is.factor(y)) levels(y) else sort(unique(y))
  msg <- paste0("Response variable is not a factor.\nAutomatically transform y to factor with following order: ",
                paste0(y_label, collapse = ","),"\n")
  if (!is.factor(y)) warning(msg)
  y <- as.factor(dense_rank(y))   # turn y to integer starts from 1
  if (resample=="bootstrapping"){
    if (stratified){
      for (i in 1:fold) {
        set.seed(seed+i, kind = "L'Ecuyer-CMRG")
        idxs[[i]] <- foreach(j = 1:k, .combine = c) %do%      
          sample(which(y == j),floor(sum(y == j)*(1-partition_ratio)))
        }
      } else {
      test_cnt <- floor((1-partition_ratio)*n)
      set.seed(seed, kind = "L'Ecuyer-CMRG")
      idxs <- lapply(1:fold, function(idx) sample(n,test_cnt))

  }
} else{  # CV
  if (stratified){
    set.seed(seed, kind = "L'Ecuyer-CMRG")
    idx <- lapply(1:k, function(i) {
      neworder <- sample(which(y==i))       # reorder indices in each class
      split(neworder, f=rep(1:fold,length.out=length(neworder))) # split to folds
    })   
    idxs <- lapply(1:fold,function(i) {     # turn to testing indexes
      c(unlist(sapply(idx,function(sub) sub[[i]])))   # c() & unlist to turn idx into vector
    })
    } else {
    set.seed(seed, kind = "L'Ecuyer-CMRG")
    idx <- sample(rep(1:fold, length.out = n))        # resample idx with "fold" kinds at length n
    idxs <- lapply(1:fold, function(i) which(idx==i)) # 
  }
}
  if (verbose==T){    # report progress
    cat("Data splitting complete.\nSettings: stratified =", stratified,", method =", resample,", classifier =", classifier,"\n")        
  }
  tr_matrix <- matrix(nrow=3*fold, ncol=k)    # training error rates
  te_matrix <- matrix(nrow=3*fold, ncol=k)    # testing error rates
  cms <- lapply(1:3, function(x) {            # confusion matrix
    mt = matrix(0,k,k) ; rownames(mt) = 1:k   # temporarily label for combining matrices
    return(mt)
  })  
  for (i in 1:fold){
    if (verbose==T & i%%5==0){    # modeling progress
      cat("current progress: split", i, "of method", classifier,"\n")        
    }
    idx <- idxs[[i]]
    train_x <- x[-idx,] ; test_x <- x[idx,]
    train_y <- y[-idx] ; test_y <- y[idx]
    cx_try <- try(npcs(x=train_x, y=train_y, algorithm="CX", classifier=classifier , 
                       w=w, alpha=alpha, trControl=trControl, tuneGrid=tuneGrid, seed=seed+1),
                  silent=TRUE)  # suppress error message
    er_try <- try(npcs(x=train_x, y=train_y, algorithm="ER", classifier=classifier , 
                       w=w, alpha=alpha, trControl=trControl, tuneGrid=tuneGrid, seed=seed+1),
                  silent=TRUE)
    
    if (!(inherits(cx_try, "try-error"))){
      cx <- cx_try
      vanilla <- cx_try$fit  # use the vanilla model in NPCS
    } else {  
      cx <- NULL
      vanilla <- modeling(data=data.frame(train_x, y=train_y), classifier=classifier, 
                          trControl=trControl, tuneGrid=tuneGrid)
    }
    if (!(inherits(er_try, "try-error"))){
      er <- er_try
    } else {
      er <- NULL}
    models <- list(vanilla=vanilla, cx=cx, er=er) 
    train_predict <- lapply(names(models), function(m) {
      if (m=="vanilla"){      # predict.train
        predict(models[[m]])
      } else if ( !(is.null(models[[m]])) ) {
        predict(models[[m]],newx=train_x)
      }
    })
    test_predict <- lapply(names(models), function(m){
      if (m=="vanilla"){  
        predict(models[[m]],newdata=data.frame(test_x))
      } else if (!(is.null(models[[m]])) ) {
        predict(models[[m]],newx=test_x)
      }
    }) 
    for (j in 1:3){
      pred_y = test_predict[[j]]
      if (!is.null(pred_y)){                  # if model feasible
        cur_t <- table(pred_y, test_y) ; cm <- cms[[j]]
        if (nrow(cur_t) != k) {                  # < k predicted classes
          fillin <- matrix(0, k-nrow(cur_t), k)  # avoid different dim
          rownames(fillin) <- rownames(cm)[!(rownames(cm) %in% rownames(cur_t))]
          cur_t <- rbind(cur_t, fillin)
          cur_t <- cur_t[order(rownames(cur_t)),]
        }
        cms[[j]] <- cm + cur_t   # confusion matrix
      }
    }
    train_error <- t(sapply(train_predict, USE.NAMES = T,simplify = T,function(p){
      if (!is.null(p)) error_rate(p,y=train_y) else rep(NA,k)
    })) 
    test_error <- t(sapply(test_predict, USE.NAMES = T,simplify = T,function(p){
      if (!is.null(p)) error_rate(p,y=test_y) else rep(NA,k)
    }))
    tr_matrix[(3*i-2):(3*i),] <- train_error
    te_matrix[(3*i-2):(3*i),] <- test_error
  }
  algorithm = c("vanilla", "cx", "er")
  output <- lapply(list(tr_matrix,te_matrix), function(mx){ # avg. error rates
    d <- data.frame(mx,
               algorithm=algorithm, 
               fold=rep(c(1:fold), each=3)) %>%
      mutate(algorithm=factor(algorithm, levels=c("vanilla", "cx","er")))
    colnames(d)[1:k] <- y_label
    return(d)
  })
  names(output) <- c("training_error","testing_error")
  # visualization:
  if (plotit){
    ti <- paste0(classifier," with ",fold,"-fold ",resample)
    pt <- output$testing_error %>%
      pivot_longer(cols = all_of(y_label), names_to = "class", values_to = "error_rate") %>%
      mutate(algorithm=fct_relevel(algorithm, "vanilla", "cx", "er")) %>%
      ggplot(aes(x=algorithm,y=error_rate, col=class)) +
      geom_violin(position = position_dodge(0.5)) +
      geom_boxplot(width=0.1, position = position_dodge(0.5)) +
      sapply(1:length(alpha), function(i) {           # threshold lines
        {if(!is.na(alpha[i])) geom_hline(yintercept = alpha[i], col=i+1)}
      }) + ggtitle(ti)
    # print(pt)
    output[["plot"]] <- pt
  }
  # outputs:
  output[["testing_indices"]] <- idxs          # index of testing data in each gp
  means <- lapply(output[1:2],function(d){
    avg <- d %>%
      group_by(algorithm) %>%
      summarize_at(vars(all_of(y_label)), mean, na.rm=T) %>%
      ungroup()
    avg[,-1]
  })
  feasibility <- aggregate(output$training_error[,1], 
                           by=list(output$training_error$algorithm),
                           FUN = function(x) mean(!is.na(x)))[,2]
  cms <- lapply(1:3, function(i) {    # avg of confusion matrix
    if (feasibility[i]!=0){
      cm <- cms[[i]]/(feasibility[i]*fold)
      rownames(cm) <- y_label
      cm
    } else{
      cms[[i]]
    }
  })  
  names(cms) <- c("vanilla","cx","er")
  summaries <- cbind(algorithm=names(cms), means[[1]],means[[2]],feasibility,
                     sapply(cms, model_eval) %>% t()
                     )
  colnames(summaries)[2:(1+2*k)] <- paste(rep(c("training","testing"), each=k), y_label,sep="_")
  output[["summaries"]] <- summaries
  output[["confusion_matrix"]] <- cms
  if (stratified) {   # each fold has same sample size in each group
    training <- table(train_y) ; testing <- table(test_y)
  } else {            # use the Avg of all folds as sample size
    testing <- colSums(cms[[1]])
    training <- table(y)-test_y
  }
  rownames(training) <- rownames(testing) <- y_label
  output[["sample_size"]] <- list(training,testing)
  class(output) <- "cv.npcs"
  return(output)
  } # end of code



  