# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Assignment: group paper
# Course: STAT 571
# Date: 2023-06-03
# Authors: Xunlei, Erin, Raul, Sarai, Huong 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
library(Matrix)
library(matrixcalc)
library(ggplot2)
library(tidyverse)
rm(list = ls())
set.seed(571)


#------------#
# parameters #
#------------#
N <- 150  # number of trajectories (i.e. people)
Q <- 4    # number of clusters (i.e. trajectory type) [we get to know this when simulating]
D <- 6    # number of measurements per trajectory
z <- rep(1:Q, ceiling(N/Q))[1:N] # cluster assignment of each trajectory (i.e. person)

beta <- c(0.1, 0.6, -.2, 1.1,-1.2,-1.7, 8. )  #<-as many as unique(Q) values
sig2 <- c(0.1, 0.2, 0.3, 0.2, 0.6, 0.3, 0.4)  #<-as many as unique(Q) values
tau2 <- c(0.9, 0.5, 0.7, 0.2, 0.2, 0.4, 0.6)  #<-as many as unique(Q) values
phi  <- 1:D


#-----------#
# fake data #
#-----------#
n <- rep(1:N, each=D)                                #<-think of it as trajectory id
e <- rep(sqrt(sig2[z]), each=D) * rnorm(n=N*D,0,1)   #<-disturbance at each time period per trajectory 
b <- rep(rnorm(N,0,sqrt(tau2[z])), each=D)           #<-random effect per cluster
t <- rep(1:D, N)                                     #<-time period
y <- rep(beta[z], each=D)*t + b + e                  #<-eq(1) on paper: Phi%*%Beta + sigma*epsilon
q <- rep(z, each=D)                                  #<-cluster the trajectory belongs to

df <- data.frame(t=t, y=y, b=b, q=q, n=n)            #<-pack it all in a dataset
df[1:20,]

mypalette <- c("#D55E00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7", "#0072B2", "#E69F00")
df |> 
  ggplot(aes(x=t, y=y, color=as.factor(q), group=as.factor(n))) +
  geom_line() +
  theme_bw() + 
  scale_colour_manual(values=mypalette) + 
  theme(legend.position = "none") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) 

# The idea: 
# We don't know what cluster each trajectory belongs to. In fact, we don't even know 
# how many clusters there are. This algorithm will help us determine cluster assignment 
# and number of clusters.
# How? We iterate over our guess of # of clusters and compute the optimal Z assignment
# at each Q=q guess based on when the algorithm stops switching trajectories to different 
# cluster q. Exact ICL is then use to assess out fit (I think). 

#----------------#
# Classification #
#----------------#
# This is basically the sum of two terms: 
# first term is:  log{P(Y|Z,Q,theta)}  
# second term is: log{P(Z|Q)}

# So, what we do is that we start with a predetermine set of number of clusters 
# (e.g., Q = 3 to 9). Then we iterate over each q, and compute the ICL. The ICL will tell 
# us the optimal Z at that Q value and the "loglikelihood" value. 
# Then we move on to the next q and compute the ICL. At the end of all q's, we'll have 
# a list of "loglikelihood" values with their respective Z assignment. The highest is our 
# chosen set of (Z,Q).  


#------------#
# First term #
#------------# 

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
  q <- unique(z)
  q_rho <- expand.grid(q=q, rho=rhos)
  y <- matrix(y, nrow=N, ncol=D, byrow = TRUE) #<-turn y into wide matrix
  
  # get likelihood per q,rho combo
  llik_q_rho <- purrr::map2_dbl(.x=q_rho$q, .y=q_rho$rho, 
                                ~llik_Y_q(y[z==.x,], rho=.y))
  sum(llik_q_rho)/N_rho
}

#-------------#
# Second term #
#-------------# 
get_logT2 <- function(alpha=10, z, Q_guess){  #D, N, Q_guess) { #<-alpha=10 is what the paper recommends
  y <- matrix(y, nrow=N, ncol=D, byrow = TRUE)
  C <- table(z) # how many individuals are in each cluster
  # from the paper we have: 
  # T2 = gamma(sum_q^Q{alpha_q})/prod_q^Q gamma(alpha) * 
  #      prod_1^Q gamma(C_q+alpha)/gamma(N + sum_1^Q alpha)
  # taking log(T2) yields: 
  lgamma(alpha*Q_guess) - Q_guess*lgamma(alpha) + 
    sum(lgamma(C + alpha)) - lgamma(N + Q_guess*alpha)
}

#-----------------------------#
# Get the assignment vector Z #
#-----------------------------#
find_Z <- function(y, z_start, Q_guess, N_rho, D, N, phi, llthresh=.01) {
  
  Q <- 1:Q_guess
  
  # z_curr and llik_curr track current state after last full iteration through i
  # z_new and llik_new track fully current state
  z_curr  <- z_start
  z_new   <- z_curr
  llik_curr <- get_logT1(y, z=z_curr, N_rho) + get_logT2(alpha=10, z_curr, Q_guess)
  llik_new <- llik_curr 
  done <- FALSE; log_diff <- 1
  print(paste0("initial likelihood = ", llik_curr))
  
  llik_afterallN <- llik_curr #<-initial loglikelihood
  while(!done){
    for(i in 1:N) {
      if(sum(z_new==z_new[i]) > 1) { # more than one individual in a cluster
        for(q in setdiff(1:Q_guess, z_curr[i])){
          z_prop <- ifelse((1:N) != i, z_new, q) # move one individual to a new cluster
          if(any((table(z_prop) |> unname()) < 2) == TRUE) {
            print("skip iteration because there's only 1 n in cluster")
            next
          }
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
    print(paste0("likelihood = ", round(llik_new, 2), 
                 "  reduction: ", round(log_diff,3)))
    done <- all(z_curr==z_new) | (log_diff < llthresh)
    llik_curr <- llik_new
    z_curr <- z_new
  }
  o <- list(z_fit=z_curr, loglikelihood=llik_curr)
  return(o)
}

# ~~~~~~~~~~~~~~~~~~~~ #
# BAYESIAN CLUSTER ME! #
# ~~~~~~~~~~~~~~~~~~~~ #
Q_guess <- 6
ZQs <- dplyr::tibble(TrajectoryN = 1:N, TrueZ = df$q[df$t==1])
llikes <- vector(mode = "numeric", length = 0)
for(q_ in 2:Q_guess) {
  z_start <- sample(rep(1:q_, ceiling(N/q_)), N) # randomly allocates Z evenly to clusters
  print(paste0("guessing ", q_, "clusters"))
  Z_fit <- find_Z(df$y, z_start=z_start, Q_guess=q_, N_rho=10, 
                  D=D, N=N, phi=phi, llthresh=.1)
  ZQs[[paste0("ZQ",q_)]] <- Z_fit$z_fit
  llikes[q_-1] <- Z_fit$loglikelihood
  #llikes[q_-1] <- get_logT1(y, z=Z_fit$z_fit, N_rho=10) + get_logT2(alpha=10, Z_fit$z_fit, Q_guess)
}
bestQ <- (2:Q_guess)[which.max(llikes)]
print(paste0("This algorithm picks Q=", bestQ))
output <- list(Z_fits = ZQs, llikes = llikes, bestQ = bestQ)

