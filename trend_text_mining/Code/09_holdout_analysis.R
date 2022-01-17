##~~~~~~~~~~~~
## Analysis of holdout set and figure plotting
##~~~~~~~~~~~~

rm(list = ls())
graphics.off()

# Load libraries

library(ggplot2)
library(caret)
library(circlize)


##~~~~~
## Functions

## Condense the 10 undersample preds into 1 consensus prediction
condense_preds <- function(df, classifier_type){
    df$Stable <- df$Increase <- df$Decline <- df$Prediction <- NA
    pred_cols <- df[,grep(paste0(classifier_type,"_1$"), colnames(df)):grep(paste0(classifier_type,"_10$"), colnames(df))]
    for (i in 1:nrow(df)){
        df$Increase[i] <- sum(pred_cols[i,] == "Increase")
        df$Stable[i] <- sum(pred_cols[i,] == "Stable")
        df$Decline[i] <- sum(pred_cols[i,] == "Decline")
        
        n_ls <- c(df$Increase[i], df$Stable[i], df$Decline[i])
        names(n_ls) <- c("Increase", "Stable", "Decline")
        
        out <- names(n_ls)[which(n_ls == max(n_ls))]
        ## How to deal with ties - -stable?
        if (length(out)>1){
            out <- "Stable"
        }
        df$Prediction[i] <- out
    }    
    
    return(df)
}


# Overall approach
# Given df of LPD, Auto and MLD categorisations calc relevant comparison metrics
perf_metr_calc <- function(df){
    
    # Prediction is Automated tool
    # Trend_sig_005_maj is from LPD
    # trend_ is from MLD
    
    # Only use levels in comprison...
    auto_lpd_lev <- sort(unique(c(df$Prediction, df$Trend_sig_005_maj)))
    man_lpd_lev <- sort(unique(c(df$trend_, df$Trend_sig_005_maj)))
    auto_man_lev <- sort(unique(c(df$Prediction, df$trend_)))
    
    auto_lpd <- confusionMatrix(data = factor(df$Prediction, levels = auto_lpd_lev),
                                reference = factor(df$Trend_sig_005_maj, levels = auto_lpd_lev))
    
    man_lpd <- confusionMatrix(data = factor(df$trend_, levels = man_lpd_lev),
                               reference = factor(df$Trend_sig_005_maj, levels = man_lpd_lev))
    
    auto_man <- confusionMatrix(data = factor(df$Prediction, levels = auto_man_lev),
                                reference = factor(df$trend_, levels = auto_man_lev))
    
    summ_df <- as.data.frame(rbind(auto_lpd$overall,
                                   man_lpd$overall,
                                   auto_man$overall))
    summ_df$comp <- c("auto_lpd", "man_lpd", "auto_man")
    
    
    return(list(auto_lpd = auto_lpd,
                man_lpd  = man_lpd,
                auto_man = auto_man,
                summ_df  = summ_df,
                
                auto_lpd_lev = auto_lpd_lev,
                man_lpd_lev  = man_lpd_lev,
                auto_man_lev = auto_man_lev
    ))
}

# Given performance metrics and an output filepath, make and save chord diagrams 
# comparing Auto, LPD and MLD trends
chord_plttr <- function(perf_metr, f_pth){
    
    # incr - blue
    # stable - amber
    # decr - red
    # varied - black
    # unclear - grey
    grid.col1a = c(T_Decline = "firebrick2", T_Stable = "#F6BB42", T_Increase = "#0072B2", T_Unclear = "darkgrey", T_Varied = "black",
                   P_Decline = "firebrick2", P_Stable = "#F6BB42", P_Increase = "#0072B2", P_Unclear = "darkgrey", P_Varied = "black")
    
    # first, define unique categories in each sub plot aand overall
    auto_lpd_t <- paste("T", perf_metr$auto_lpd_lev, sep = "_")
    auto_lpd_p <- paste("P", perf_metr$auto_lpd_lev, sep = "_")
    
    man_lpd_t <- paste("T", perf_metr$man_lpd_lev, sep = "_")
    man_lpd_p <- paste("P", perf_metr$man_lpd_lev, sep = "_")
    
    auto_man_t <- paste("T", perf_metr$auto_man_lev, sep = "_")
    auto_man_p <- paste("P", perf_metr$auto_man_lev, sep = "_")
    
    # sort ordering
    cat_ord <- c("P_Increase","P_Stable", "P_Decline", "P_Varied", "P_Unclear",
                 "T_Unclear", "T_Varied", "T_Decline", "T_Stable", "T_Increase")
    # drop missing...
    
    # fig legend ...
    leg_ord <- c("Increase", "Stable", "Decline", "Varied", "Unclear")
    
    
    leg_col <- c("#0072B2", "#FFCE54", "firebrick2", "black", "darkgrey")
    
    # Make plot,..
    
    pdf(f_pth, width = 15, height = 6)
    
    par(mfrow= c(1,3), oma = c(0,0,0,5), xpd = NA)
    
    # font = 2
    
    # A. LPD v Auto
    circos.clear()
    circos.par(start.degree = 90, 
               canvas.xlim = c(-1.2, 1.2), 
               canvas.ylim = c(-1, 1.25))
    chordDiagram(data.frame(
        to = rep(auto_lpd_t, each = length(auto_lpd_t)),
        from = rep(auto_lpd_p, times = length(auto_lpd_t)),
        value = c(perf_metr$auto_lpd$table)),
        grid.col = grid.col1a,
        annotationTrack = c("grid"),
        # link.sort = T, link.decreasing = F,
        order = cat_ord[cat_ord %in% c(auto_lpd_t, auto_lpd_p)]
    )
    
    segments(x0 = 0, y0 = -1, y1 = 1, lty = 2, col = "#00000080")
    text(x = -1.15, y = 1.15, pos = 4, labels = "a)", cex = 2.5, font = 2)
    text(x = 1, y = 1, labels = "Automated", pos = 2, cex = 2.3)
    text(x = -1, y = 1, labels = "Full text", pos = 4, cex = 2.3)
    text(x = -1, y = -1, labels = bquote(paste(kappa, ": ", 
                                               .(as.character(round(as.numeric(perf_metr$auto_lpd$overall[["Kappa"]]), 3))),
                                               sep="")), 
         pos = 4, cex = 2.1)
    text(x = 1, y = -1, labels = bquote(paste("Acc: ", 
                                              .(as.character(round(as.numeric(perf_metr$auto_lpd$overall[["Accuracy"]])*100, 1))),
                                              "%",
                                              sep="")), 
         pos = 2, cex = 2.1)
    
    # B. MLD v Auto
    circos.clear()
    circos.par(start.degree = 90,
               canvas.xlim = c(-1.2, 1.2), 
               canvas.ylim = c(-1, 1.25))
    chordDiagram(data.frame(
        to = rep(auto_man_t, each = length(auto_man_t)),
        from = rep(auto_man_p, times = length(auto_man_t)),
        value = c(perf_metr$auto_man$table)),
        grid.col = grid.col1a,
        annotationTrack = c("grid"),
        # link.sort = T, link.decreasing = F,
        order = cat_ord[cat_ord %in% c(auto_man_t, auto_man_p)]
    )
    
    segments(x0 = 0, y0 = -1, y1 = 1, lty = 2, col = "#00000080")
    text(x = -1.15, y = 1.15, labels = "b)", pos = 4, cex = 2.5, font = 2)
    text(x = 1, y = 1, labels = "Automated", pos = 2, cex = 2.3)
    text(x = -1, y = 1, labels = "Abstract", pos = 4, cex = 2.3)
    text(x = -1, y = -1, labels = bquote(paste(kappa, ": ", 
                                               .(as.character(round(as.numeric(perf_metr$auto_man$overall[["Kappa"]]), 3))),
                                               sep="")), 
         pos = 4, cex = 2.1)
    text(x = 1, y = -1, labels = bquote(paste("Acc: ", 
                                              .(as.character(round(as.numeric(perf_metr$auto_man$overall[["Accuracy"]])*100, 1))),
                                              "%",
                                              sep="")), 
         pos = 2, cex = 2.1)
    
    # Place legend at bottom of central sub plt
    
    legend(
        x = "bottom",
        inset = -0.035,
        pch = 15,
        bty = "n",
        horiz = T,
        
        title = "Trend category",
        col = leg_col[leg_ord %in% perf_metr$auto_man_lev],
        legend = leg_ord[leg_ord %in% perf_metr$auto_man_lev],
        cex = 2.1)
    
    # C. LPD v MLD
    circos.clear()
    circos.par(start.degree = 90,
               canvas.xlim = c(-1.2, 1.2), 
               canvas.ylim = c(-1, 1.25))
    chordDiagram(data.frame(
        to = rep(man_lpd_t, each = length(man_lpd_t)),
        from = rep(man_lpd_p, times = length(man_lpd_t)),
        value = c(perf_metr$man_lpd$table)),
        grid.col = grid.col1a,
        annotationTrack = c("grid"),
        # link.sort = T, link.decreasing = F,
        order = cat_ord[cat_ord %in% c(man_lpd_t, man_lpd_p)]
    )
    # abline(v = 0, lty = 2, col = "#00000080")
    segments(x0 = 0, y0 = -1, y1 = 1, lty = 2, col = "#00000080")
    text(x = -1.15, y = 1.15, labels = "c)", pos = 4, cex = 2.5, font = 2)
    text(x = 1, y = 1, labels = "Abstract", pos = 2, cex = 2.3)
    text(x = -1, y = 1, labels = "Full text", pos = 4, cex = 2.3)
    text(x = -1, y = -1, labels = bquote(paste(kappa, ": ", 
                                               .(as.character(round(as.numeric(perf_metr$man_lpd$overall[["Kappa"]]), 3))),
                                               sep="")), 
         pos = 4, cex = 2.1)
    text(x = 1, y = -1, labels = bquote(paste("Acc: ", 
                                              .(as.character(round(as.numeric(perf_metr$man_lpd$overall[["Accuracy"]])*100, 1))),
                                              "%",
                                              sep="")), 
         pos = 2, cex = 2.1)
    
    
    dev.off()
    
}


##~~~~~


##%%%%%
## Main Code

# Load auto model df
rf_df <- read.csv("../Results/holdout_preds/rf1.csv", stringsAsFactors = F)

# Condense 10 predictions - 1 per seed, into 1, majority rule.
rf_df <- condense_preds(rf_df, "rf")
# Sensitivty in predictions to sampling of training data!!


# Load manual categorisations
man_class_df <- read.csv("../Data/compiled_trend_text1.csv", 
                         encoding = "latin1",
                         stringsAsFactors = F)
# And LPD trends
lpi_trends <- read.csv("../Data/lpi_trends.csv")

man_class_comp <- merge(man_class_df[,c("RN", "trend", "multi_trend_ind")],
                        lpi_trends)

# Adjust manual clasifications where necessary 
man_class_comp$trend_ <- as.character(man_class_comp$trend)
man_class_comp$trend_[man_class_comp$trend_ == "stable"] <- "Stable"
man_class_comp$trend_[man_class_comp$trend_ == "decrease"] <- "Decline"
man_class_comp$trend_[man_class_comp$trend_ == "decline"] <- "Decline"
man_class_comp$trend_[man_class_comp$trend_ == "increase"] <- "Increase"
man_class_comp$trend_[man_class_comp$trend_ == "varied"] <- "Varied"
man_class_comp$trend_[man_class_comp$trend_ == "vareid"] <- "Varied"
man_class_comp$trend_[man_class_comp$trend_ == "unclear"] <- "Unclear"

# Issue: manual classes include "Varied", issue as text model cannot predict this...
# Means that model v human is going to differ anyway....


##
## Combine Auto, Manual and LPD trend categories for the 300 MLD texts
plt_df <- merge(rf_df[,c("RN", "Trend_sig_005_maj", "Prediction")],
                man_class_comp[,c("RN", "Trend_sig_005_maj", "trend_")])

unique(plt_df$Trend_sig_005_maj)
unique(plt_df$Prediction)
unique(plt_df$trend_)

# given automated cannot predict varied trends, drop these from the 300...
plt_df_sub <- subset(plt_df, Trend_sig_005_maj != "Varied")

## In addition to this, also further subset to where manual classification 
## is not Varied or unclear
## Allows for fairer comparison between approches
plt_df_sub1 <- subset(plt_df, Trend_sig_005_maj != "Varied" & !(trend_ %in% c("Unclear", "Varied")))

# Only 111 texts 

# For each of full plt_df (300 MLD), plt_df_sub (279, no true Varied), and 
# plt_df_sub1 (111, all incr/decr/stab), calculate performance metrics
perf_metr <- perf_metr_calc(plt_df)
perf_metr_sub <- perf_metr_calc(plt_df_sub)
perf_metr_sub1 <- perf_metr_calc(plt_df_sub1)

## Quickly lok at perf metrics

# Full MLD results
perf_metr$summ_df

perf_metr$auto_lpd$overall
perf_metr$man_lpd$overall
perf_metr$auto_man$overall


# Results based on only texts where truth is not varied
perf_metr_sub$summ_df

perf_metr_sub$auto_lpd$overall
perf_metr_sub$man_lpd$overall
perf_metr_sub$auto_man$overall


# Results based on only texts where all categorisations are in incr/decl/stab
perf_metr_sub1$summ_df

perf_metr_sub1$auto_lpd$overall
perf_metr_sub1$man_lpd$overall
perf_metr_sub1$auto_man$overall


##%%%%
# Plotting
# Make and save plots
chord_plttr(perf_metr, "../Results/Figs/chord_diag.pdf")
# chord_plttr(perf_metr_sub, "../Results/Figs/chord_diag_sub.pdf")
chord_plttr(perf_metr_sub1, "../Results/Figs/chord_diag_sub1.pdf")



# Some quick summaries of trend categoristions
perf_metr_sub1$auto_lpd
perf_metr_sub1$man_lpd$table

2/(24+2+8) 
2/(2+25+5)
# v few mis class of incr:decl

# Number of varied and unclear texts
sum(plt_df$Trend_sig_005_maj == "Varied")
sum(plt_df$trend_ == "Varied")
sum(plt_df$trend_ == "Unclear")

sum(plt_df$trend_ == "Varied")+sum(plt_df$trend_ == "Unclear")

# Representation of predictions (Auto and MLD) vs LPD
table(plt_df_sub1$Prediction)/table(plt_df_sub1$Trend_sig_005_maj)
table(plt_df_sub1$trend_)/table(plt_df_sub1$Trend_sig_005_maj)

table(plt_df_sub1$Prediction)/table(plt_df_sub1$trend_)

# Check of automated method miss-classifying increases as declines and vice versa
sum(plt_df_sub1$Prediction == "Decline" & plt_df_sub1$Trend_sig_005_maj == "Increase")
sum(plt_df_sub1$Prediction == "Increase" & plt_df_sub1$Trend_sig_005_maj == "Decline")
sum(plt_df_sub1$Trend_sig_005_maj == "Increase")
sum(plt_df_sub1$Trend_sig_005_maj == "Decline")


sum(plt_df_sub1$Prediction == "Decline" & plt_df_sub1$Trend_sig_005_maj == "Stable")
sum(plt_df_sub1$Prediction == "Increase" & plt_df_sub1$Trend_sig_005_maj == "Stable")
sum(plt_df_sub1$Trend_sig_005_maj == "Stable")


