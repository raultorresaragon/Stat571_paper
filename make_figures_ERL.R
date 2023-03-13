# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Assignment: group paper
# Course: STAT 571
# Date: 2023-06-03
# Authors: Xunlei, Erin, Raul, Sarai, Huong 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
source("simulation_function_ERL.R")

library(tidyverse)
library(ggplot2)
library(pander)

res <- readRDS(file.path("res","combined.RDS"))

summy <- res %>%
  mutate(Q = map(res, ~.x[[3]]),
         Z = map(res, ~.x[[1]]),
         llik = map(res, ~.x[[2]])) %>%
  select(-res) %>%
  unnest(c(Q)) %>%
  mutate(iter=rep(1:(nrow(res)/2), each=2))

# sample trajectories
beta <- c(.5, 0, -.5)
sig2 <- c(0.05, 0.1, 0.3)
tau2 <- c(0.1, 0.1, 0.25)
set.seed(5711)
df <- gen_data(Q=3, beta, sig2, tau2,D=4,N=50)
plot_dat(df)
ggsave(file.path("figs","sample.jpeg"))   

#summy %>% ggplot(aes(x=iter, y=Q, color=label)) +
#  geom_point()

# Q
summy %>%
  count(label, Q) %>%
  pivot_wider(names_from=label, values_from=n) %>%
  mutate(corr=ifelse(is.na(corr),0,corr)) %>%
  pivot_longer(-Q, names_to="label", values_to="n") %>%
  group_by(label) %>%
  mutate(p=n/sum(n)) %>%
  ggplot(aes(x=Q, y=p, fill=label)) +
  geom_col(position = position_dodge(width=1)) +
  geom_text(aes(label=round(p,2)),
            position=position_dodge(width=1), vjust=-.5) +
  ylim(0,1.01) +
  theme_bw()
ggsave(file.path("figs","which_Q.jpeg"))   

summy %>%
  select(label, iter, Q, llik) %>%
  mutate(q=list(2:5)) %>%
  unnest(c(llik,q)) %>%
  mutate(k = q*(2+as.numeric(label=="corr")),
         BIC = k*log(50)-2*llik,
         llik=-2*llik) %>%
  pivot_longer(c("BIC","llik"), 
               names_to="type", values_to="value") %>%
  mutate(type=ifelse(type=="llik","-2llik",type)) %>%
  ggplot(aes(x=as.factor(q), y=value, color=label)) +
  geom_boxplot() +
  facet_grid(type~., scales="free_y") +
  theme_bw()
ggsave(file.path("figs","llik.jpeg"))   

# confusion
# make_correlation <- function(Q,Zmat){
#   est <- Zmat[[paste0("ZQ",Q)]]
#   N=length(est)
#   cross_df(list(i1=1:N, i2=1:N)) %>%
#     mutate(z1=map(i1, ~est[.x]),
#            z2=map(i2, ~est[.x])) %>%
#     unnest(c(z1,z2)) %>% 
#     mutate(p=z1==z2) %>%
#     select(i1,i2,p)
#}

make_confusion <- function(Q,Zmat){
  est <- Zmat[[paste0("ZQ",Q)]]
  true <- Zmat$TrueZ
  
  # remap est for identifiability
  est_map <- rep(NA, Q)
  for (i in 1:Q){
    most_common <- sort(table(est[true==i]), decreasing = T)
    if (length(most_common)==0){
      i2 <- setdiff(1:Q,est_map)[1]
    } else {
      i2 <- names(most_common[!names(most_common) %in% est_map])[1]
    }
    est_map[i] <- i2  
  }
  conf <- table(est,true)[est_map,]
  rownames(conf) <- 1:Q
  as_tibble(conf)
}

confusion <- select(summy, iter, label, Q, Z) %>%
  filter(Q==3) %>%
  mutate(confusion=map2(Q,Z,make_confusion)) %>%
  select(iter, label, confusion) %>%
  unnest(confusion) %>%
  group_by(label, est, true) %>%
  summarise(n=sum(n)) %>%
  group_by(label,true) %>%
  mutate(p=n/sum(n))

#order <- map(1:3, ~which(summy$Z[[1]]$TrueZ==.x)) %>% unlist()
#order_map <- tibble(old = order, new=1:N)
confusion %>%
  filter(est==true) %>%
  ggplot(aes(x=true, y=1-p, fill=label)) +
  geom_col(position=position_dodge()) +
  geom_text(aes(label=round(1-p,3)),
            position=position_dodge(width=1), vjust=-.5) +
  xlab("True cluster") +
  ylab("Missclassification rate") + ylim(0,.085) +
  theme_minimal()
ggsave(file.path("figs","missclassification.jpeg"))   
