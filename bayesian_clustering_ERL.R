# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Assignment: group paper
# Course: STAT 571
# Date: 2023-06-03
# Authors: Xunlei, Erin, Raul, Sarai, Huong 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
source("simulation_function_ERL.R")

library(tidyverse)
library(ggplot2)
set.seed(571)

# generate fake data
N <- 50   # number of trajectories (i.e. people)
Q <- 3    # number of clusters (i.e. trajectory type) [we get to know this when simulating]
D <- 4    # number of measurements per trajectory
phi <- 1:D

beta <- c(.5, 0, -.5)
sig2 <- c(0.05, 0.1, 0.3)
tau2 <- c(0.1, 0.1, 0.25)

tau2/(sig2+tau2)

# ~~~~~~~~~~~~~~~~~~~~ #
# BAYESIAN CLUSTER ME! #
# ~~~~~~~~~~~~~~~~~~~~ #
fit_one <- function(df, N_rho = 10, Q_guess = 5){
  ZQs <- dplyr::tibble(TrajectoryN = 1:N, TrueZ = df$q[df$t==1])
  llikes <- vector(mode = "numeric", length = 0)
  for(q_ in 2:Q_guess) {
    z_start <- kmeans(matrix(df$y,ncol=D,byrow = T), q_)$cluster
    print(paste0("guessing ", q_, " clusters"))
    Z_fit <- find_Z(df$y, z_start=z_start, Q_guess=q_, N_rho=N_rho, 
                    D=D, N=N, phi=phi, llthresh=0)
    ZQs[[paste0("ZQ",q_)]] <- Z_fit$z_fit
    llikes[q_-1] <- Z_fit$loglikelihood
  }
  bestQ <- (2:Q_guess)[which.max(llikes)]
  print(paste0("This algorithm picks Q=", bestQ))
  output <- list(Z_fits = ZQs, llikes = llikes, bestQ = bestQ)  
}

sim_one <- function(iter = 1){
  set.seed(10*571 + iter)
  df <- gen_data(3, beta, sig2, tau2,D=D,N=N)
  
  fit_corr <- fit_one(df, 10)
  fit_ind <- fit_one(df, 1)
  
  out <- tibble(label = c("corr","ind"),
                res = list(fit_corr, fit_ind))
  saveRDS(out, file.path("res",paste0("res_", iter, ".RDS")))
  return(out)
}

combine_res <- function(iter=1:10){
  files <- paste0("res_", iter, ".RDS")
  comb <- map_df(file.path("res",files), readRDS)
  saveRDS(comb, file.path("res","combined.RDS"))
}

# run!
for (iter in 31:50){
  print(paste0("iter ", iter))
  out <- sim_one(iter)
  print(paste0("Q corr: ", out$res[[1]]$bestQ,
               " | Q ind: ", out$res[[2]]$bestQ))
}

combine_res(1:50)

#df %>% mutate(q=rep(ZQs$ZQ3, each=D)) %>% plot_dat()
#table(ZQs$TrueZ, ZQs$ZQ3)

#z <- ZQs$TrueZ
#get_logT1(df$y, z, N_rho=1) + get_logT2(10,z,2)
#z <- ZQs$ZQ3
#get_logT1(df$y, z, N_rho=1) + get_logT2(10,z,2)
              
