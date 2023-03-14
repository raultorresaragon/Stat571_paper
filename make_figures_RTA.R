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
set.seed(5711)
mypalette <- c("#D55E00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#0072B2", "#E69F00")

# ~~~~~~~~~~~~~~ #
# Summarize data #
# ~~~~~~~~~~~~~~ #
res <- readr::read_rds("combined.RDS")
summy <- res |>
  mutate(Q = map(res, ~.x[[3]]),
         Z = map(res, ~.x[[1]]),
         llik = map(res, ~.x[[2]])) |>
  dplyr::select(-res) |>
  unnest(c(Q)) |>
  mutate(iter=rep(1:(nrow(res)/2), each=2))


# ~~~~~~~~~~~~~~~~~~~ #
# sample trajectories #
# ~~~~~~~~~~~~~~~~~~~ #
plot_traject <- function(df, mypalette){
    g <- df |> 
         ggplot(aes(x=t, y=y, color=as.factor(q), group=as.factor(n))) +
         geom_line() +
         theme_bw() + 
         xlab("D") + 
         scale_colour_manual(name='Cluster (Q)', 
                             values=mypalette[1:3],
                             labels=c("q=1","q=2","q=3")) +
         theme(panel.grid.major = element_blank(), 
               panel.grid.minor = element_blank(),
               axis.text=element_text(size=12),
               axis.title=element_text(size=14),
               legend.position = "right",
               legend.title = element_text(size=14),
               legend.text  = element_text(size=12)) 
    plot(g)
}

beta <- c(.5, 0, -.5)
sig2 <- c(0.05, 0.1, 0.3)
tau2 <- c(0.1, 0.1, 0.25)
df <- gen_data(Q=3, beta, sig2, tau2, D=4, N=50)
plot_traject(df, mypalette)
ggsave(file.path("figs","sample.jpeg"))   



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Proportion selecting each Q, bar chart #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
myplot_df <- summy |>
             count(label, Q) |>
             tidyr::pivot_wider(names_from=label, values_from=n) |>
             dplyr::mutate(corr=ifelse(is.na(corr),0,corr)) |>
             tidyr::pivot_longer(-Q, names_to="label", values_to="n") |>
             group_by(label) |>
             dplyr::mutate(p=n/sum(n))
p_left <- myplot_df |>
          ggplot(aes(x=Q, y=p, fill=label)) +
          geom_col(position = position_dodge(width=1)) +
          geom_text(aes(label=round(p,2)),
                    position=position_dodge(width=1), vjust=-.5) +
          ylim(0,1.01) + 
          scale_fill_manual(name='Correlation\n  Structure', 
                            values = mypalette[1:2]) + 
          theme_bw() + 
          ylab("Proportion Q choice") + 
          theme(panel.grid.major = element_blank(), 
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                axis.text.y=element_blank(),
                axis.ticks.y=element_blank(),
                axis.text=element_text(size=12),
                axis.title=element_text(size=14))
p_left
ggsave(file.path("figs","which_Q.jpeg"))   

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Missclassification rate bar chart #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
make_confusion <- function(Q, Zmat){
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

confusion <- dplyr::select(summy, iter, label, Q, Z) |>
             dplyr::filter(Q==3) |>
             dplyr::mutate(confusion=map2(Q,Z,make_confusion)) |>
             dplyr::select(iter, label, confusion) |>
             tidyr::unnest(confusion) |>
             dplyr::group_by(label, est, true) |>
             dplyr::summarise(n=sum(n)) |>
             dplyr::group_by(label,true) |>
             dplyr::mutate(p=n/sum(n)) |> 
             dplyr::filter(est==true)
             

p_right <- confusion |>
  ggplot(aes(x=true, y=1-p, fill=label)) +
  geom_col(position=position_dodge()) +
  geom_text(aes(label=round(1-p,3)),
            position=position_dodge(width=1), vjust=-.5) +
  xlab("True Q") +
  ylab("Missclassification rate") + ylim(0,.085) + 
  scale_fill_manual(name='Correlation\n  Structure', values = mypalette[1:2]) +
  theme_minimal() + 
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        axis.text=element_text(size=12),
        axis.title=element_text(size=14))
p_right
ggsave(file.path("figs","whichQ_AND_missclass.jpeg"))   

ggpubr::ggarrange(
  p_left, p_right, labels = NULL,
  common.legend = TRUE, legend = "top"
)
ggsave(file.path("figs","missclassification.jpeg")) 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Boxplots BIC and -2 LogLikelihood #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
myplot_df <- summy |>
             dplyr::select(label, iter, Q, llik) |>
             dplyr::mutate(q=list(2:5)) |>
             tidyr::unnest(c(llik,q)) |>
             dplyr::mutate(k = q*(2+as.numeric(label=="corr")),
                           BIC = k*log(50)-2*llik,
                           llik=-2*llik) |>
             tidyr::pivot_longer(c("BIC","llik"), 
                                 names_to="type", 
                                 values_to="value") |>
             dplyr::mutate(type=ifelse(type=="llik","-2llik",type)) 

myplot_df |> 
  ggplot(aes(x=as.factor(q), y=value, fill=label)) +
  geom_boxplot() + 
  xlab("Q") + 
  scale_fill_manual(name='Correlation\n  Structure', 
                    values = mypalette[1:2]) + 
  facet_grid(type~., scales="free_y") +
  theme_bw() + 
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.text=element_text(size=12),
        axis.title=element_text(size=14),
        legend.position = "right",
        legend.title = element_text(size=14),
        legend.text  = element_text(size=12))
ggsave(file.path("figs","llik.jpeg"))   
