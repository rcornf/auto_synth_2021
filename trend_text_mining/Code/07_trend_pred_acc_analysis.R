##~~~~~~~~~~~~
## Analysis of lpi trend classification models
##~~~~~~~~~~~~

rm(list = ls())
graphics.off()

# Load libraries
# library(tidyverse)
library(plyr)
library(ggplot2)
library(cowplot)
library(caret)
library(dplyr)
library(readr)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Functions

# ggplot theme
basic_thm <- theme(axis.text = element_text(size = 16),
                   axis.title = element_text(size = 20),
                   plot.title = element_text(size = 20),
                   plot.subtitle = element_text(size = 18),
                   strip.text.x = element_text(size = 18),
                   strip.text.y = element_text(size = 18),
                   legend.text = element_text(size = 16),
                   legend.title = element_text(size = 18),
                   legend.text.align = 0) 

# String to vector
# Function to convert string of numbers to vector
str_to_vector <- function(input_str){
    # Remove [ and ] and ,
    str <- gsub("\\[|\\]|,|'", "", input_str)
    
    # Separate by ' '
    vect <- unlist(strsplit(str, " ")[[1]])
    
    return (vect)
}

# Calculate accuracy and kappa for individual model runs
acc_calc <- function(df){

    tr_true <- c()
    tr_pred <- c()
    te_true <- c()
    te_pred <- c()
    
    tr_acc_vec <- c()
    te_acc_vec <- c()
    
    # extract true and predicted classes, 
    for (row in 1:nrow(df)){
        tmp_tr_true <- str_to_vector(df$Train_class[row])
        tmp_tr_pred <- str_to_vector(df$Train_pred[row])
        tmp_te_true <- str_to_vector(df$Test_class[row])
        tmp_te_pred <- str_to_vector(df$Test_pred[row])
        
        tr_true <- c(tr_true, tmp_tr_true)
        tr_pred <- c(tr_pred, tmp_tr_pred)
        te_true <- c(te_true, tmp_te_true)
        te_pred <- c(te_pred, tmp_te_pred)
        
        # Per fold acc
        tr_acc_vec <- c(tr_acc_vec, sum(tmp_tr_pred == tmp_tr_true)/length(tmp_tr_pred))
        te_acc_vec <- c(te_acc_vec, sum(tmp_te_pred == tmp_te_true)/length(tmp_te_pred))
    }
    
    # Accuracy over all folds
    tr_acc <- sum(tr_pred == tr_true)/length(tr_pred)
    te_acc <- sum(te_pred == te_true)/length(te_pred)
    # Average per fold accuracy
    tr_acc_mn <- mean(tr_acc_vec)
    te_acc_mn <- mean(te_acc_vec)
    
    conf_matr <- confusionMatrix(data = factor(te_pred),
                                 reference = factor(te_true))
    
    N_tot <- df$N_train[1]+df$N_test[1]
    # return df of specs plus metrics...
    out <- data.frame("tr_acc" = tr_acc,
                      "te_acc" = te_acc,
                      "tr_acc_mn" = tr_acc,
                      "te_acc_mn" = te_acc,
                      "te_acc_conf" = as.vector(conf_matr$overall["Accuracy"]),
                      "te_acc_conflo" = as.vector(conf_matr$overall["AccuracyLower"]),
                      "te_acc_confhi" = as.vector(conf_matr$overall["AccuracyUpper"]),
                      "te_kappa" = as.vector(conf_matr$overall["Kappa"]),
                      "n_tot" = N_tot)
}


# Summarise kappa and aaccuracy over the 10 undersmple repeats per classifier type
kappa_acc_summ <- function(x){
    kap_quants <- quantile(x$te_kappa, c(0,0.05,0.25,0.5,0.75,0.975,1))
    acc_quants <- quantile(x$te_acc, c(0,0.05,0.25,0.5,0.75,0.975,1))
    
    x$kap_min <- kap_quants[[1]]
    x$kap_lo_95 <- kap_quants[[2]]
    x$kap_lo <- kap_quants[[3]]
    x$kap_med <- kap_quants[[4]]
    x$kap_hi <- kap_quants[[5]]
    x$kap_hi_95 <- kap_quants[[6]]
    x$kap_max <- kap_quants[[7]]
    
    x$acc_min <- acc_quants[[1]]
    x$acc_lo_95 <- acc_quants[[2]]
    x$acc_lo <- acc_quants[[3]]
    x$acc_med <- acc_quants[[4]]
    x$acc_hi <- acc_quants[[5]]
    x$acc_hi_95 <- acc_quants[[6]]
    x$acc_max <- acc_quants[[7]]
    
    return(x[1,])
}

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##%%%%%
## RF initial tests

# Load data
rf_df <- list.files(path="../Results/rf_init", full.names = TRUE) %>% 
    lapply(read_csv) %>% 
    bind_rows 

# Calculate accuracy and kappa
rf_acc <- plyr::ddply(rf_df, 
                   c("id", "Class_type", "Classifier", "Incl_varied", "Seed"), 
                   acc_calc)

# These cat types don't have any varied trends
rf_acc <- subset(rf_acc, !((Class_type == "Trend_meta_005" |
                         Class_type == "Trend_meta_01") &
                         Incl_varied == T))

# Summarise accuracy results
rf_summ <- ddply(rf_acc, .(Class_type, Classifier, Incl_varied), kappa_acc_summ)

rf_summ$Class_type <- factor(rf_summ$Class_type,
                             levels = rev(c("Trend_01_maj", "Trend_02_maj", "Trend_05_maj",  
                                        "Trend_01_60", "Trend_02_60", "Trend_05_60",             
                                        "Trend_sig_005_maj", "Trend_sig_01_maj",
                                        "Trend_sig_005_60", "Trend_sig_01_60",
                                        "Trend_meta_005", "Trend_meta_01")))

# Identify those models that appear best
rf_summ$Selected <- "No"
rf_summ$Selected[(grepl("sig", rf_summ$Class_type) &
                     rf_summ$Incl_varied == F)] <- "Yes"
## Plots
ggplot(rf_summ) +
    geom_errorbar(aes(x = Class_type,
                      ymin = kap_lo_95, ymax = kap_hi_95,
                      colour = Selected),
                  width = 0,
                  size = 1) +
    geom_errorbar(aes(x = Class_type,
                     ymin = kap_lo, ymax = kap_hi,
                     colour = Selected),
                  width = 0,
                  size = 2) +
    geom_point(aes(x = Class_type, y = kap_med,
                   colour = Selected),
               size = 2.75) +
    labs(x = "Categorisation appraoch",
         y = "Test kappa") +
    scale_colour_manual(values = c("No" = "black",
                                   "Yes" = "firebrick3"),
                        guide = F) +
    theme_bw() +
    basic_thm +
    coord_flip() +
    facet_wrap(~ Incl_varied,
               labeller = labeller(Incl_varied = c("FALSE" = "Excl. varied",
                                                   "TRUE"  = "Incl. varied"))) +
    theme(#axis.text.x = element_text(angle = 60, hjust = 1),
          strip.background = element_blank(), 
          panel.grid.minor = element_blank())


##
ggplot(rf_summ) +
    geom_errorbar(aes(x = Class_type,
                      ymin = acc_lo_95, ymax = acc_hi_95,
                      colour = Selected),
                  width = 0,
                  size = 1) +
    geom_errorbar(aes(x = Class_type,
                      ymin = acc_lo, ymax = acc_hi,
                      colour = Selected),
                  width = 0,
                  size = 2) +
    geom_point(aes(x = Class_type, y = acc_med,
                   colour = Selected),
               size = 2.75) +
    labs(x = "Categorisation appraoch",
         y = "Test accuracy") +
    scale_colour_manual(values = c("No" = "black",
                                   "Yes" = "firebrick3"),
                        guide = F) +
    theme_bw() +
    basic_thm +
    coord_flip() +
    facet_wrap(~ Incl_varied,
               labeller = labeller(Incl_varied = c("FALSE" = "Excl. varied",
                                                   "TRUE"  = "Incl. varied"))) +
    theme(#axis.text.x = element_text(angle = 60, hjust = 1),
        strip.background = element_blank(), 
        panel.grid.minor = element_blank())

# order of x
# magn., [0.1, 0.2, 0.5], [maj, 60], 
# sig, [005, 01], [maj, 60]
# meta [005, 01]

# Class_type to retain: Trend_sig_005_60,Trend_sig_005_maj,Trend_sig_01_60,Trend_sig_01_maj 

##%%%%%
## NN initial testing - using the categorisation based on prop of sig trends in text
# Load data
nn_df <- list.files(path="../Results/nn_init", full.names = TRUE) %>% 
    lapply(read_csv) %>% 
    bind_rows 
# Calculate accuracy
nn_acc <- plyr::ddply(nn_df, 
                   c("id", "Class_type", "Classifier", "Incl_varied", "Seed"), 
                   acc_calc)

nn_acc <- subset(nn_acc, Incl_varied == F)
# Summarise accuracy
nn_summ <- ddply(nn_acc, .(Class_type, Classifier, Incl_varied), kappa_acc_summ)

## Plots
ggplot(nn_summ) +
    geom_errorbar(aes(x = Class_type,
                      ymin = kap_lo_95, ymax = kap_hi_95),
                  width = 0,
                  size = 1) +
    geom_errorbar(aes(x = Class_type,
                      ymin = kap_lo, ymax = kap_hi),
                  width = 0,
                  size = 2) +
    geom_point(aes(x = Class_type, y = kap_med),
               size = 2.75) +
    labs(x = "Categorisation appraoch",
         y = "Test kappa") +
    # scale_colour_manual(values = c("No" = "black",
    #                                "Yes" = "firebrick3"),
    #                     guide = F) +
    theme_bw() +
    basic_thm +
    coord_flip() +
    # facet_wrap(~ Incl_varied,
    #            labeller = labeller(Incl_varied = c("FALSE" = "Excl. varied",
    #                                                "TRUE"  = "Incl. varied"))) +
    theme(#axis.text.x = element_text(angle = 60, hjust = 1),
        strip.background = element_blank(), 
        panel.grid.minor = element_blank())



# No masive boost in perf....
nn_summ$Selected <- "Yes"

##%%%%%
# Combine rf and nn init results into a single fig...
init_summ <- rbind(rf_summ, nn_summ)
init_summ$Classifier <- factor(init_summ$Classifier,
                               levels = c("rf", "nn"))
# Better trend categorisation names
# sig., mag.
init_summ$Class_type_ <- as.character(init_summ$Class_type)

init_summ$discr <- "Mag."
init_summ$discr[grepl("sig", init_summ$Class_type_)] <- "Sig."
init_summ$discr[grepl("meta", init_summ$Class_type_)] <- "Metafor"

init_summ$agg <- ""
init_summ$agg[grepl("maj", init_summ$Class_type_)] <- "a."
init_summ$agg[grepl("60", init_summ$Class_type_)] <- "b."

init_summ$alph <- 0.01
init_summ$alph[grepl("_005", init_summ$Class_type_)] <- 0.05
init_summ$alph[grepl("_05", init_summ$Class_type_)] <- 0.05
init_summ$alph[grepl("_02", init_summ$Class_type_)] <- 0.02

init_summ$Class_type__ <- paste(init_summ$discr, init_summ$alph, init_summ$agg)
init_summ$Class_type__ <- factor(init_summ$Class_type__,
                                 c("Metafor 0.01 ", "Metafor 0.05 ",
                                   "Sig. 0.05 b.", "Sig. 0.05 a.", 
                                   "Sig. 0.01 b.", "Sig. 0.01 a.",
                                   "Mag. 0.05 b.", "Mag. 0.05 a.",
                                   "Mag. 0.02 b.", "Mag. 0.02 a.",
                                   "Mag. 0.01 b.", "Mag. 0.01 a."
                                   ))

# Plot of initial models and highlighting those that have been selected
init_summ_plt <- ggplot(init_summ) +
    geom_errorbar(aes(x = Class_type__,
                      ymin = kap_lo_95, ymax = kap_hi_95,
                      colour = Selected),
                  width = 0,
                  size = 1) +
    geom_errorbar(aes(x = Class_type__,
                      ymin = kap_lo, ymax = kap_hi,
                      colour = Selected),
                  width = 0,
                  size = 2) +
    geom_point(aes(x = Class_type__, y = kap_med,
                   colour = Selected),
               size = 2.75) +
    labs(x = "Categorisation appraoch",
         y = "Kappa") +
    scale_colour_manual(values = c("No" = "black",
                                   "Yes" = "firebrick3"),
                        guide = F) +
    theme_bw() +
    basic_thm +
    coord_flip() +
    facet_grid(Classifier~ Incl_varied,
               scales = "free_y",
               drop = TRUE,
               labeller = labeller(Incl_varied = c("FALSE" = "Excl. varied",
                                                   "TRUE"  = "Incl. varied"),
                                   Classifier = c("nn" = "Neural network",
                                                  "rf"  = "Random forest"))) +
    theme(#axis.text.x = element_text(angle = 60, hjust = 1),
        strip.background = element_blank(), 
        panel.grid.minor = element_blank())

ggsave("../Results/Figs/rfnn_init_plt.pdf",
       init_summ_plt,
       device = "pdf",
       dpi = 300,
       width = 9, height = 7)



##%%%%%
## Analysis of se (within-paper trend variation) impact -- i.e, does reducing
## the within paper trend variation improve trend prediction

# Load data
rfnn_se_df <- list.files(path="../Results/rfnn_se", full.names = TRUE) %>% 
    lapply(read_csv) %>% 
    bind_rows 

# Calculate and then summarise model performance
rfnn_se_acc <- plyr::ddply(rfnn_se_df, 
                       c("id", "Class_type", "Classifier", "se_t", "Seed"), 
                       acc_calc)

rfnn_se_summ <- ddply(rfnn_se_acc, .(Class_type, Classifier, se_t), kappa_acc_summ)

rfnn_se_summ$Classifier <- factor(rfnn_se_summ$Classifier,
                                   levels = c("rf", "nn"))
rfnn_se_summ$type <- "se"

# Get selected models from initial tests
init_summ_sub <- subset(init_summ, Selected == "Yes")
init_summ_sub$Class_type__ <- factor(init_summ_sub$Class_type__,
                                     c("Sig. 0.01 a.", "Sig. 0.01 b.",
                                       "Sig. 0.05 a.", "Sig. 0.05 b."
                                     ))

# Fix names for plotting
rfnn_se_summ$discr <- "Sig."

rfnn_se_summ$agg <- "a."
rfnn_se_summ$agg[grepl("60", rfnn_se_summ$Class_type)] <- "b."

rfnn_se_summ$alph <- 0.01
rfnn_se_summ$alph[grepl("_005", rfnn_se_summ$Class_type)] <- 0.05

rfnn_se_summ$Class_type__ <- paste(rfnn_se_summ$discr, rfnn_se_summ$alph, rfnn_se_summ$agg)
rfnn_se_summ$Class_type__ <- factor(rfnn_se_summ$Class_type__,
                                 c("Sig. 0.01 a.", "Sig. 0.01 b.",
                                   "Sig. 0.05 a.", "Sig. 0.05 b."
                                 ))

# Plot to compare intial models with those adjusting within-paper trend variation
se_comp_plt <- plot_grid(ggplot(rfnn_se_summ) +
              geom_line(aes(x = se_t, y = kap_med, colour = Class_type__), size = 1) +
              geom_point(aes(x = se_t, y = kap_med, colour = Class_type__), size = 2) +
              ylim(c(0.175,0.325)) +
              labs(x = "SE threshold",
                   y = "Kappa") +
              theme_bw() +
              basic_thm +
              facet_grid(Classifier~.,
                         labeller = labeller(Classifier = c("nn" = "Neural network",
                                                            "rf" = "Random forest"))) +
              theme(strip.background = element_blank(),
                    strip.text.y = element_blank(),
                    legend.position = "none"),
          
          ggplot(init_summ_sub) +
              geom_point(aes(x = 0, y = kap_med, colour = Class_type__), size = 3) +
              ylim(c(0.175,0.325)) +
              labs(x = "All texts",
                   y = "",
                   colour = "Categorisation\napproach") +
              theme_bw() +
              basic_thm +
              facet_grid(Classifier~.,
                         labeller = labeller(Classifier = c("nn" = "Neural network",
                                                            "rf" = "Random forest"))) +
              theme(strip.background = element_blank(),
                    axis.text = element_blank(),
                    axis.ticks.x = element_blank(),
                    panel.grid.minor.x = element_blank()),
          align = "hv"#,
          # labels = c("a.", "b."), label_size = 18
          )
se_comp_plt

ggsave("../Results/Figs/rfnn_se_plt.pdf",
       se_comp_plt,
       device = "pdf",
       dpi = 300,
       width = 9, height = 7)


##%%%%%
## Augmentation analysis -- testing ipmacts of different text augmentation on trend
## prediction performance

## RF models
rf_aug_df <- list.files(path="../Results/rf_aug", full.names = TRUE) %>% 
    lapply(read_csv) %>% 
    bind_rows 

rf_aug_acc <- plyr::ddply(rf_aug_df, 
                             c("id", "Class_type", "Classifier", "Incl_varied", 
                               "Sent_strp", "Sp_rm", "Loc_rm", "Augmented", "Alpha_Sigma", 
                               "N_aug", "Seed"), 
                             acc_calc)

rf_aug_summ <- ddply(rf_aug_acc, .(Class_type, Classifier, Incl_varied, Sent_strp, Sp_rm, Loc_rm, Augmented, Alpha_Sigma, N_aug), kappa_acc_summ)

rf_aug_summ$Classifier <- factor(rf_aug_summ$Classifier,
                                    levels = c("rf", "nn"))

rf_aug_summ$discr <- "Sig."

rf_aug_summ$agg <- "a."
rf_aug_summ$agg[grepl("60", rf_aug_summ$Class_type)] <- "b."

rf_aug_summ$alph <- 0.01
rf_aug_summ$alph[grepl("_005", rf_aug_summ$Class_type)] <- 0.05

rf_aug_summ$Class_type__ <- paste(rf_aug_summ$discr, rf_aug_summ$alph, rf_aug_summ$agg)
rf_aug_summ$Class_type__ <- factor(rf_aug_summ$Class_type__,
                                      c("Sig. 0.01 a.", "Sig. 0.01 b.",
                                        "Sig. 0.05 a.", "Sig. 0.05 b."
                                      ))

# Compare the different effects of Sentence removal, Loction removal and Species removal
rf_aug_summ1 <- rbind.fill(subset(init_summ_sub, Classifier =="rf"),
                        subset(rf_aug_summ, Augmented == F & Loc_rm == F & Sp_rm == F &
                                   Sent_strp == T &
                                   Incl_varied == F),
                        subset(rf_aug_summ, Augmented == F & Loc_rm == T & Sp_rm == F &
                                   Sent_strp == F &
                                   Incl_varied == F),
                        subset(rf_aug_summ, Augmented == F & Loc_rm == F & Sp_rm == T &
                                   Sent_strp == F &
                                   Incl_varied == F)
)
rf_aug_summ1$Aug_app <- "Baseline"
rf_aug_summ1$Aug_app[rf_aug_summ1$Sent_strp == T] <-"Sentence removal"
rf_aug_summ1$Aug_app[rf_aug_summ1$Loc_rm == T] <-"Location removal"
rf_aug_summ1$Aug_app[rf_aug_summ1$Sp_rm == T] <-"Species removal"

rf_aug_summ1$Aug_app <- factor(rf_aug_summ1$Aug_app,
                            levels = c("Baseline", "Sentence removal",
                                       "Species removal", "Location removal")) 

rf_aug1_plt <- ggplot(rf_aug_summ1) +
    geom_errorbar(aes(x = 0,
                      ymin = kap_lo_95, ymax = kap_hi_95,
                      colour = Class_type__),
                  position = position_dodge(0.5),
                  width = 0,
                  size = 1) +
    geom_errorbar(aes(x = 0,
                      ymin = kap_lo, ymax = kap_hi,
                      colour = Class_type__),
                  position = position_dodge(0.5),
                  width = 0,
                  size = 2) +
    geom_point(aes(x = 0, y = kap_med, colour = Class_type__), 
               position = position_dodge(0.5),
               size = 3) +
    # ylim(c(0.175,0.325)) +
    labs(x = "",
         y = "Kappa",
         colour = "Categorisation\napproach") +
    theme_bw() +
    basic_thm +
    facet_grid(Classifier~Aug_app,
               switch="x",
               labeller = labeller(Classifier = c("nn" = "Neural network",
                                                  "rf" = "Random forest"))) +
    theme(strip.background = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          panel.grid.minor.x = element_blank())
rf_aug1_plt


# Compare the impact of using the EDA library
rf_aug_summ2 <- rbind.fill(subset(init_summ_sub, Classifier == "rf"),
                        subset(rf_aug_summ, Augmented == T & Alpha_Sigma == 0.05 &
                                   Loc_rm == F & Sp_rm == F &
                                   Sent_strp == F &
                                   Incl_varied == F))

rf_aug_summ2$N_aug[is.na(rf_aug_summ2$Augmented)] <- 0

rf_aug2_plt <- ggplot(rf_aug_summ2) +
    geom_line(aes(x = N_aug, y = kap_med, colour = Class_type__),
               # position = position_dodge(0.5),
               size = 1) +
    geom_point(aes(x = N_aug, y = kap_med, colour = Class_type__),
              # position = position_dodge(0.5),
              size = 2) +
    labs(x = "Number of augmented texts",
         y = "Kappa",
         colour = "Categorisation\napproach") +
    theme_bw() +
    basic_thm +
    facet_grid(Classifier~.,
               labeller = labeller(Classifier = c("nn" = "Neural network",
                                                  "rf" = "Random forest"))) +
    theme(strip.background = element_blank(),
          # axis.text.x = element_blank(),
          # axis.ticks.x = element_blank(),
          panel.grid.minor.x = element_blank())
rf_aug2_plt

##


## NN models
nn_aug_df <- list.files(path="../Results/nn_aug", full.names = TRUE) %>% 
    lapply(read_csv) %>% 
    bind_rows 

nn_aug_acc <- plyr::ddply(nn_aug_df, 
                          c("id", "Class_type", "Classifier", "Incl_varied", 
                            "Sent_strp", "Sp_rm", "Loc_rm", "Augmented", "Alpha_Sigma", 
                            "N_aug", "Seed"), 
                          acc_calc)

nn_aug_summ <- ddply(nn_aug_acc, .(Class_type, Classifier, Incl_varied, Sent_strp, Sp_rm, Loc_rm, Augmented, Alpha_Sigma, N_aug), kappa_acc_summ)

nn_aug_summ$Classifier <- factor(nn_aug_summ$Classifier,
                                 levels = c("rf", "nn"))
# nn_aug_summ$Class_type

nn_aug_summ$discr <- "Sig."

nn_aug_summ$agg <- "a."
nn_aug_summ$agg[grepl("60", nn_aug_summ$Class_type)] <- "b."

nn_aug_summ$alph <- 0.01
nn_aug_summ$alph[grepl("_005", nn_aug_summ$Class_type)] <- 0.05

nn_aug_summ$Class_type__ <- paste(nn_aug_summ$discr, nn_aug_summ$alph, nn_aug_summ$agg)
nn_aug_summ$Class_type__ <- factor(nn_aug_summ$Class_type__,
                                   c("Sig. 0.01 a.", "Sig. 0.01 b.",
                                     "Sig. 0.05 a.", "Sig. 0.05 b."
                                   ))

# note, where n_aug == 16 and class type is sig 01 maj, need to drop as too few model runs worked...
nn_aug_summ_sub <- subset(nn_aug_summ, !(Class_type == "Trend_sig_01_maj" & N_aug == 16) | is.na(N_aug))

##
# Compare the different effects of Sentence removal, Loction removal and Species removal
nn_aug_summ1 <- rbind.fill(subset(init_summ_sub, Classifier =="nn"),
                           subset(nn_aug_summ_sub, Augmented == F & Loc_rm == F & Sp_rm == F &
                                      Sent_strp == T &
                                      Incl_varied == F),
                           subset(nn_aug_summ_sub, Augmented == F & Loc_rm == T & Sp_rm == F &
                                      Sent_strp == F &
                                      Incl_varied == F),
                           subset(nn_aug_summ_sub, Augmented == F & Loc_rm == F & Sp_rm == T &
                                      Sent_strp == F &
                                      Incl_varied == F)
)
nn_aug_summ1$Aug_app <- "Baseline"
nn_aug_summ1$Aug_app[nn_aug_summ1$Sent_strp == T] <-"Sentence removal"
nn_aug_summ1$Aug_app[nn_aug_summ1$Loc_rm == T] <-"Location removal"
nn_aug_summ1$Aug_app[nn_aug_summ1$Sp_rm == T] <-"Species removal"

nn_aug_summ1$Aug_app <- factor(nn_aug_summ1$Aug_app,
                               levels = c("Baseline", "Sentence removal",
                                          "Species removal", "Location removal")) 

ggplot(nn_aug_summ1) +
    geom_errorbar(aes(x = 0,
                      ymin = kap_lo_95, ymax = kap_hi_95,
                      colour = Class_type__),
                  position = position_dodge(0.5),
                  width = 0,
                  size = 1) +
    geom_errorbar(aes(x = 0,
                      ymin = kap_lo, ymax = kap_hi,
                      colour = Class_type__),
                  position = position_dodge(0.5),
                  width = 0,
                  size = 2) +
    geom_point(aes(x = 0, y = kap_med, colour = Class_type__), 
               position = position_dodge(0.5),
               size = 3) +
    # ylim(c(0.175,0.325)) +
    labs(x = "",
         y = "Kappa",
         colour = "Categorisation\napproach") +
    theme_bw() +
    basic_thm +
    facet_grid(Classifier~Aug_app,
               switch="x",
               labeller = labeller(Classifier = c("nn" = "Neural network",
                                                  "rf" = "Random forest"))) +
    theme(strip.background = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          panel.grid.minor.x = element_blank())


# Compare the impact of using the EDA library
nn_aug_summ2 <- rbind.fill(subset(init_summ_sub, Classifier == "nn"),
                           subset(nn_aug_summ_sub, Augmented == T & Alpha_Sigma == 0.05 &
                                      Loc_rm == F & Sp_rm == F &
                                      Sent_strp == F &
                                      Incl_varied == F))

nn_aug_summ2$N_aug[is.na(nn_aug_summ2$Augmented)] <- 0

ggplot(nn_aug_summ2) +
    geom_line(aes(x = N_aug, y = kap_med, colour = Class_type__),
              # position = position_dodge(0.5),
              size = 1) +
    geom_point(aes(x = N_aug, y = kap_med, colour = Class_type__),
               # position = position_dodge(0.5),
               size = 2) +
    labs(x = "Number of augmented texts",
         y = "Kappa",
         colour = "Categorisation\napproach") +
    theme_bw() +
    basic_thm +
    facet_grid(Classifier~.,
               labeller = labeller(Classifier = c("nn" = "Neural network",
                                                  "rf" = "Random forest"))) +
    theme(strip.background = element_blank(),
          # axis.text.x = element_blank(),
          # axis.ticks.x = element_blank(),
          panel.grid.minor.x = element_blank())

##

##%%%%%
# Combine augmented dfs to make a single plot comparing impacts of augmentation 
# initial/baseline models

rfnn_aug_summ1 <- rbind(rf_aug_summ1,
                        nn_aug_summ1)

rfnn_aug_summ2 <- rbind(rf_aug_summ2,
                        nn_aug_summ2)

## Plot
rfnn_aug_plt <- plot_grid(ggplot(rfnn_aug_summ2) +
    geom_line(aes(x = N_aug, y = kap_med, colour = Class_type__),
              # position = position_dodge(0.5),
              size = 1) +
    geom_point(aes(x = N_aug, y = kap_med, colour = Class_type__),
               # position = position_dodge(0.5),
               size = 2) +
    labs(x = "Number of augmented texts",
         y = "Kappa",
         colour = "Categorisation\napproach") +
        ylim(c(0.18,0.39)) +
    theme_bw() +
    basic_thm +
    facet_grid(Classifier~.,
               labeller = labeller(Classifier = c("nn" = "Neural network",
                                                  "rf" = "Random forest"))) +
    theme(strip.background = element_blank(),
          # axis.text.x = element_blank(),
          # axis.ticks.x = element_blank(),
          # legend.position = "none",
          panel.grid.minor.x = element_blank()),
    NULL,
    
    ggplot(rfnn_aug_summ1) +
    geom_errorbar(aes(x = 0,
                      ymin = kap_lo_95, ymax = kap_hi_95,
                      colour = Class_type__),
                  position = position_dodge(0.5),
                  width = 0,
                  size = 1) +
    geom_errorbar(aes(x = 0,
                      ymin = kap_lo, ymax = kap_hi,
                      colour = Class_type__),
                  position = position_dodge(0.5),
                  width = 0,
                  size = 2) +
    geom_point(aes(x = 0, y = kap_med, colour = Class_type__), 
               position = position_dodge(0.5),
               size = 3) +
    ylim(c(0.18,0.39)) +
    labs(x = "",
         y = "Kappa",
         colour = "Categorisation\napproach") +
    theme_bw() +
    basic_thm +
    facet_grid(Classifier~Aug_app,
               switch="x",
               labeller = labeller(Classifier = c("nn" = "Neural network",
                                                  "rf" = "Random forest"))) +
    theme(strip.background = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.position = "none",
          panel.grid.minor.x = element_blank()), 
    align = "hv",
    rel_heights = c(0.5,0.1,0.5), 
    nrow = 3,
    labels = c("a.", "","b."), label_size = 18
)

ggsave("../Results/Figs/rfnn_aug_plt.pdf",
       rfnn_aug_plt,
       device = "pdf",
       dpi = 300,
       width = 10, height = 10)
###

