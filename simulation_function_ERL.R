library(Matrix)
library(matrixcalc)
library(tidyverse)
library(cluster)
set.seed(571)

# helper functions

gen_data <- function(Q,beta,sig2,tau2,
                     D=5,N=50){
  z <- rep(1:Q, ceiling(N/Q))[1:N] # cluster assignment of each trajectory (i.e. person)
  n <- rep(1:N, each=D)                                #<-think of it as trajectory id
  e <- rep(sqrt(sig2[z]), each=D) * rnorm(n=N*D,0,1)   #<-disturbance at each time period per trajectory 
  b <- rep(rnorm(N,0,sqrt(tau2[z])), each=D)           #<-random effect per cluster
  t <- rep(1:D, N)                                     #<-time period
  y <- rep(beta[z], each=D)*t + b + e                  #<-eq(1) on paper: Phi%*%Beta + sigma*epsilon
  q <- rep(z, each=D)                                  #<-cluster the trajectory belongs to
  
  df <- data.frame(t=t, y=y, b=b, q=q, n=n)            #<-pack it all in a dataset
  return(df)
}

plot_dat <- function(df){
  g <- df |> 
    ggplot(aes(x=t, y=y, color=as.factor(q), group=as.factor(n))) +
    geom_line() +
    theme_bw() + 
    #scale_colour_manual(values=mypalette) + 
    theme(legend.position = "none") +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())   
  plot(g)
}


# This first function gets loglikelihood for a given Q=q and Rho=rho
llik_Y_q <- function(y, rho, eta=1, a=1, b=1) {
  # from appendix E:
  # We need to compute det(G), inv(G), and quadratic form y^T G^-1 y for each q
  C <- nrow(y)
  A <- eta * phi %*% t(phi)
  R <- toeplitz(c(1, rep(rho, D-1))) #<-correlation matrix
  
  # log determinant of G
  # from Erin's calculations: det(G) = det(R)^{c-1} times det(R+cA)
  detR <- (1-rho)^(D-1)*(1+(D-1)*rho)
  ldetG <- (C-1)*log(det(R)) + log(det(as.matrix(R+C*A)))
  
  # quadratic form term yGy
  Rinv_w <- -rho/((1-rho)*(1+(D-1)*rho))
  Rinv_v <- 1/(1-rho) + Rinv_w
  Rinv   <- (1/(1-rho))*diag(D) + Rinv_w
  W <- -solve(R+C*A)%*%A%*%Rinv
  yGy <- sum(map_dbl(1:nrow(y),~(y[.x,]%*%Rinv%*%y[.x,]))) + colSums(y)%*%W%*%colSums(y)
  
  # log likelihood
  t1 <- -0.5*(D*C)*log(2*pi) - 0.5*ldetG
  t2 <- a*log(b) -lgamma(a) + lgamma(D*C/2 + a)
  t3 <- -(D*C/2 + a)*log(b+yGy/2)
  output <- t1 + t2 + t3
  return(output)
}

# Now, we integrate over multiple rho vals (we draw from beta distribution)
get_logT1 <- function(y, z, N_rho=10) {
  
  rhos <- rbeta(N_rho, 1, 2.5) #<-draw N_rho rho values from beta(1, 2.5)
  if (N_rho==1){rhos <- 0}
  q <- unique(z)
  q_rho <- expand.grid(q=q, rho=rhos)
  y <- matrix(y, nrow=N, ncol=D, byrow = TRUE) #<-turn y into wide matrix
  
  # get likelihood per q,rho combo
  llik_q_rho <- mutate(q_rho, 
                       llik = purrr::map2(.x=q_rho$q, .y=q_rho$rho,  
                                          ~llik_Y_q(y[z==.x,,drop=F], rho=.y))) %>%
    unnest(llik)
  
  group_by(llik_q_rho, q) %>%
    summarise(llik = log(mean(exp(llik)))) %>%
    pull(llik) %>% sum()
}

get_logT2 <- function(alpha=10, z, Q_guess){  #D, N, Q_guess) { #<-alpha=10 is what the paper recommends
  C <- table(z) # how many individuals are in each cluster
  # from the paper we have: 
  # T2 = gamma(sum_q^Q{alpha_q})/prod_q^Q gamma(alpha) * 
  #      prod_1^Q gamma(C_q+alpha)/gamma(N + sum_1^Q alpha)
  # taking log(T2) yields: 
  lgamma(alpha*Q_guess) - Q_guess*lgamma(alpha) + 
    sum(lgamma(C + alpha)) - lgamma(N + Q_guess*alpha)
}

find_Z <- function(y, z_start, Q_guess, N_rho, D, N, phi, llthresh=.01) {
  Q <- 1:Q_guess
  
  # z_curr and llik_curr track current state after last full iteration through i
  # z_new and llik_new track fully current state
  z_curr  <- z_start
  z_new   <- z_curr
  llik_curr <- get_logT1(y, z=z_curr, N_rho) + get_logT2(alpha=10, z_curr, Q_guess)
  llik_new <- llik_curr 
  done <- FALSE; log_diff <- 1
  #print(paste0("initial likelihood = ", llik_curr))
  
  while(!done){
    for(i in 1:N) {
      if(sum(z_new==z_new[i]) > 1) { # more than one individual in a cluster
        for(q in setdiff(1:Q_guess, z_curr[i])){
          z_prop <- ifelse((1:N) != i, z_new, q) # move one individual to a new cluster
          llik_prop <- get_logT1(y=y, z=z_prop, N_rho) + get_logT2(alpha=10, z_prop, Q_guess)
          if(llik_prop>llik_curr){
            z_new <- z_prop
            llik_new <- llik_prop
          }
        }
      }
    }
    # Check whether llik has changed by more than threshold, then update llik
    log_diff <- llik_new-llik_curr
    #print(paste0("likelihood = ", round(llik_new, 2), 
    #             "  reduction: ", round(log_diff,3),
    #             "  changes: ", sum(z_curr!=z_new)))
    done <- all(z_curr==z_new) | (log_diff < llthresh)
    llik_curr <- llik_new
    z_curr <- z_new
  }
  o <- list(z_fit=z_curr, loglikelihood=llik_curr)
  print(paste0("likelihood = ", llik_curr))
  return(o)
}


