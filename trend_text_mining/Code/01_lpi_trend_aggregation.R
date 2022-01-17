#####
## Rscript to calculate meta-trends for each data source
## Using meta-regression code from TJ
#####


# clear env
rm(list = ls())
graphics.off()

# Load packages
library(metafor)
library(ggplot2)
library(plyr)

# Load lpi data
lpi_dat <- read.csv("../Data/lpi_pops_RN_202009010.csv")

# lpi_dat <- subset(lpi_dat, DP_total > 2)

# colnames(lpi_dat)
# X1950-X2018, [76:144]
lpi_dat$lmbd1 <- lpi_dat$lmbd2 <- lpi_dat$vr1 <- lpi_dat$vr2 <- NA 
lpi_dat$lmbd1_lo<- lpi_dat$lmbd1_hi <- lpi_dat$lmbd2_lo <- lpi_dat$lmbd2_hi <- NA

# For each row, calc lm of log(abund)/log10(abund)
for (i in 1:nrow(lpi_dat)){
    tmp_df <- data.frame("Year" = 1950:2018,
                         "abund" = unname(t(lpi_dat[i,76:144])))
    tmp_df <- na.omit(tmp_df)
    
    # fix pop vals if 0s
    tmp_df$abund[tmp_df$abund==0] <- 0.01*mean(tmp_df$abund[tmp_df$abund!=0])
    # +1 if pop vals < 1
    if (sum(tmp_df$abund<1)>0){
        tmp_df$abund <- tmp_df$abund + 1
    }
    
    tmp_df$log10_abund <- log10(tmp_df$abund)
    tmp_df$log_abund <- log(tmp_df$abund)
    
    tmp_m1 <- lm(log10_abund ~ Year, data = tmp_df)
    tmp_m2 <- lm(log_abund ~ Year, data = tmp_df)
    
    lpi_dat$lmbd1[i] <- coef(tmp_m1)[2]
    lpi_dat$lmbd2[i] <- coef(tmp_m2)[2]
    
    lpi_dat$vr1[i] <- vcov(tmp_m1)[2,2]
    lpi_dat$vr2[i] <- vcov(tmp_m2)[2,2]
    
    ci1 <- confint(tmp_m1)
    ci2 <- confint(tmp_m2)
    
    lpi_dat$lmbd1_lo[i] <- ci1[2,1]
    lpi_dat$lmbd1_hi[i] <- ci1[2,2]
    lpi_dat$lmbd2_lo[i] <- ci2[2,1]
    lpi_dat$lmbd2_hi[i] <- ci2[2,2]
    
    lpi_dat$lmbd1_p[i] <- summary(tmp_m1)$coefficients[2,4]
    lpi_dat$lmbd2_p[i] <- summary(tmp_m2)$coefficients[2,4]
}


sum(is.na(lpi_dat$vr1))
sum(is.na(lpi_dat$vr2))

# Issue of NA variance/CIs when only 2 data-points
# Simple fix for variance issues...
# Where n == 2, set var to high..., 0.2
lpi_dat$vr1[lpi_dat$DP_total == 2] <- 0.2
lpi_dat$vr2[lpi_dat$DP_total == 2] <- 0.2
# Where n > 2, but var zero, set to low, 0.000001
lpi_dat$vr1[(lpi_dat$vr1 == 0) & (!is.na(lpi_dat$vr1))] <- 0.000001
lpi_dat$vr2[(lpi_dat$vr2 == 0) & (!is.na(lpi_dat$vr2))] <- 0.000001



# Trend categorisation methods...
# Significance of trends - ideally using metafor..
# Magnitude of trend - if multi, use majority/ if over 60% - as in current work
# Also, for magnitude - use metafor beta

# Individual pop trend classes
# threshold-based 
trend_class_t <- function(lambda_series, threshold){
    out <- rep("Stable", length(lambda_series))
    
    out[(lambda_series>=threshold)] <- "Increase"
    out[(lambda_series<= -1*threshold)] <- "Decline"
    return(out)
}	

# significance-based
trend_class_sig <- function(lambda_series, sig_series, sig_t){
    out <- rep("Stable", length(lambda_series))
    
    out[((lambda_series>0) & (sig_series<sig_t) & (!is.na(sig_series)))] <- "Increase"
    out[((lambda_series<0) & (sig_series<sig_t) & (!is.na(sig_series)))] <- "Decline"
    return(out)
}


# Aggregate trend classes
# Assign trend based on overall distr of individual trends within paper
# thresholds - majority/60%
# significance - metafor - overall trend
# significance - majority/60% 

trend_agg_60 <- function(tr_vec){ 
    n_ <- length(tr_vec)
    incr_prop <- sum(tr_vec=="Increase")/n_ 
    decl_prop <- sum(tr_vec=="Decline")/n_
    stab_prop <- sum(tr_vec=="Stable")/n_
    # Increase:
    # Incr_prop>0.6 and Incr_prop>=2*Decl_prop
    if ((incr_prop>0.6) & (incr_prop>2*decl_prop)){
        return("Increase")
    }
        
    # Decline:
    # Incr_prop>0.6 and Decl_prop>=2*Incr_prop
    else if ((decl_prop>0.6) & (decl_prop>2*incr_prop)){
        return("Decline")
    }		
        
    # Stable:
    # Stab_prop>0.6 and Stab_prop>=2*Incr_prop Stab_prop>=2*Decl_prop 
    else if ((stab_prop>0.6) & (stab_prop>2*incr_prop) & stab_prop>2*decl_prop){
        return("Stable")
    }
        
    # Varied:
    # Incr_prop>0.33 and Decl_prop>0.33
    # All others??
    else {
        return("Varied")
    }
}


trend_agg_maj <- function(tr_vec){ 
    n_ <- length(tr_vec) 
    incr_prop <- sum(tr_vec=="Increase")/n_ 
    decl_prop <- sum(tr_vec=="Decline")/n_
    stab_prop <- sum(tr_vec=="Stable")/n_
    # id highest prop ... set as class
    prop_ls <- c(incr_prop, decl_prop, stab_prop)
    names(prop_ls) <- c("Increase", "Decline", "Stable")
    
    out <- names(prop_ls)[which(prop_ls == max(prop_ls))]
    # print(out)
    # What if multiple..., set to varied...
    if (length(out)>1){
        out <- "Varied"
    }
    return(out)
}

# no class, just quant - metafor



# also note metafor se, incl based on thresholds across range of se..
# Variance of multi trend abstracts...


# a function to generate agg trends for a given type of input lambda (log v log10)
# needs to use each of thresholds - at maj and 60% level
# then merge with metafor output to and use sig to threshold...
# return df of RN, trend01_60, trend01_maj ..., trend_meta_sig01, trend_meta_sig005, trend_meta

trend_agg_wrapper <- function(df, lmbd_col, var_col){

    t_ls <- c(0.01,0.02,0.05)
    names(t_ls) <- c("Trend_01", "Trend_02", "Trend_05")
    
    trend_t_df <- data.frame(lapply(t_ls, FUN = 
                                        function(x, y){trend_class_t(y, x)}, 
                                    df[,lmbd_col]))
    
    lmbd_p_col <- paste0(lmbd_col, "_p")
    
    sig_ls <- c(0.1,0.05)
    names(sig_ls) <- c("Trend_sig_01", "Trend_sig_005")
    
    trend_sig_df <- data.frame(lapply(sig_ls, FUN = 
                                        function(x, y, z){trend_class_sig(y, z, x)}, 
                                    df[,lmbd_col],
                                    df[,lmbd_p_col]))
    
    # Also need lo, hi and p
    lmbd_lo_col <- paste0(lmbd_col, "_lo")
    lmbd_hi_col <- paste0(lmbd_col, "_hi")
    
    tmp_df <- cbind(df[,c("RN", lmbd_col, var_col, lmbd_lo_col,
                          lmbd_hi_col, lmbd_p_col)], 
                    trend_t_df,
                    trend_sig_df)

    out_df <- ddply(tmp_df, .(RN), 
                    function(x, lmbd_col, var_col, lmbd_lo_col, lmbd_hi_col, lmbd_p_col){
                        
                        # Aggregate trends based on thresholds
                        tr_60 <- as.data.frame.list(apply(x[,c("Trend_01", "Trend_02", "Trend_05")], 
                                      2, 
                                      trend_agg_60))
                        colnames(tr_60) <- c("Trend_01_60", "Trend_02_60", "Trend_05_60")
                   
                        tr_maj <- as.data.frame.list(apply(x[,c("Trend_01", "Trend_02", "Trend_05")], 
                                        2, 
                                        trend_agg_maj))
                        colnames(tr_maj) <- c("Trend_01_maj", "Trend_02_maj", "Trend_05_maj")
                        
                        # Aggregate trends based on significance
                        tr_s_60 <- as.data.frame.list(apply(x[,c("Trend_sig_01", "Trend_sig_005")], 
                                                            2, 
                                                            trend_agg_60))
                        colnames(tr_s_60) <- c("Trend_sig_01_60", "Trend_sig_005_60")
                        
                        tr_s_maj <- as.data.frame.list(apply(x[,c("Trend_sig_01", "Trend_sig_005")], 
                                                            2, 
                                                            trend_agg_maj))
                        colnames(tr_s_maj) <- c("Trend_sig_01_maj", "Trend_sig_005_maj")
                        
                        
                        # Aggregate trends based on metafor values
                        if (nrow(x)>1){
                            m_cmb <- metafor::rma(get(lmbd_col), get(var_col), data = x)
                            tr_meta <- data.frame(av_lmbd = m_cmb$beta[1],
                                             loCI = m_cmb$ci.lb,
                                             upCI = m_cmb$ci.ub,
                                             se = m_cmb$se,
                                             p = m_cmb$pval,
                                             N = m_cmb$k)
                        }
                        else {
                            tr_meta <- data.frame(av_lmbd = x[,lmbd_col],
                                             loCI = x[,lmbd_lo_col],
                                             upCI = x[,lmbd_hi_col],
                                             se = NA,
                                             p = x[,lmbd_p_col],
                                             N = 1)
                        }
                        
                        return(data.frame(cbind(tr_60, tr_maj, tr_s_60, tr_s_maj, tr_meta)))
                    },
                    lmbd_col, var_col, lmbd_lo_col, lmbd_hi_col, lmbd_p_col)
    # Significant metafor trends
    out_df$Trend_meta_01 <- trend_class_sig(out_df$av_lmbd, out_df$p, 0.1)
    out_df$Trend_meta_005 <- trend_class_sig(out_df$av_lmbd, out_df$p, 0.05)
    
    return(out_df)
}

# rm(lmbd1_agg)
lmbd_agg <- trend_agg_wrapper(lpi_dat, "lmbd1", "vr1")
write.csv(lmbd_agg, "../Data/lpi_trends.csv")
