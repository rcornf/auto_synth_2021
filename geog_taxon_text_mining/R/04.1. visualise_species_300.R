# script for checking the accuracy of species assignment from scraping

# read in packages
library(dplyr)
library(ggplot2)
library(here)
library(data.table)
library(forcats)
library(patchwork)
library(ape)
library(rotl)
library(ggtree)
library(viridis)
library(ggnewscale)
library(ggridges)
library(grid)
library(ggtreeExtra)
library(ggpubr)
library(phytools)

# get the unique sampled RN
paper_IDs <- read.csv("data/compiled_trend_text.csv") %>%
  pull(RN)

# read in the catlogue of life cleaned data
unique_col <- readRDS("data/unique_COL_species.rds")
unique_col <- unique_col %>%
  rename(scientific_name = temp) %>%
  mutate_all(as.character)

# trim unique_col white space
unique_col$scientific_name <- unique_col$scientific_name %>%
  as.character() %>%
  trimws("r")

# read in the abstract species, slect for just the lcoation column, and then filter out NA locations
abstract_sample <- read.csv("data/compiled_trend_text.csv") %>%
  filter(!is.na(species_sci)) %>%
  filter(species_sci != "")

# select unique abstract IDs and create empty list
RNs <- unique(abstract_sample$RN)
abstract_list <- list()

for(i in 1:length(RNs)){
  abstract_RN <- abstract_sample %>% filter(RN %in% RNs[i])
  abstract_list[[i]] <- data.frame("RN" = RNs[i], "species" = c(unlist(strsplit(as.character(abstract_RN$species_sci), ",|;"))))
}

# clean up abstract manual data
abstract_sample_comma <- rbindlist(abstract_list) %>% 
  mutate(species = trimws(species)) %>%
  mutate(species = stringr::word(species, 1 , 2)) %>%
  select(RN, species) %>%
  unique() %>%
  mutate(ID_col = 1:403)

# read in the actual LPI data and filter for those we sampled - remove duplicates pointed out by Rich
taxa_lpi <- read.csv(("data/lpi_pops_remove_foreign-text.csv")) %>%
  mutate(Binomial = paste(Genus, Species, sep = " ")) %>%
  select(RN, Binomial) %>%
  unique() %>%
  filter(!RN %in% c(2587, 2812, 2843))

# read in the scraped taxonomic data - remove duplicates pointed out by Rich
taxa_scraped <- read.csv(("outputs/02. post_COL_species_scrape_LPI_remove_foreign-texts.csv")) %>% 
  select(RN, original, scientific_name, taxa_data.class.i., taxa_data.order.i.) %>%
  unique() %>%
  filter(!RN %in% c(2587, 2812, 2843))

taxa_lpi = taxa_lpi[taxa_lpi$RN %in% abstract_sample_comma$RN, ]
taxa_scraped = taxa_scraped[taxa_scraped$RN %in% abstract_sample_comma$RN, ]
  
### script below working on recall i.e. number of species in each abstract correctly found
# manual LPD vs auto
cmb_df = NULL
for(a in unique(taxa_lpi$RN)){
  ref_df = subset(taxa_lpi, RN == a)
  com_df = subset(taxa_scraped, RN == a)
  if(nrow(com_df) < 1){
    
  } else {
    tally = c()
    for(b in unique(ref_df$Binomial)){
      check = as.numeric(any(com_df$scientific_name == b))
      tally = c(tally, check)
    }
    tmp_df = data.frame(
      RN = a,
      acc = (sum(tally)/length(tally))*100
    )
    cmb_df = rbind(cmb_df, tmp_df)
  }
}
mean(cmb_df$acc)
sd(cmb_df$acc)

den_spec_lpi_geo = ggplot(data = cmb_df) +
  geom_density(aes(x = acc), size = 1.2) +
  geom_vline(aes(xintercept = mean(acc)), colour = "red") +
  xlab(" ") +
  ylab("Density") +
  scale_y_continuous(expand = c(0, 0), limits = c(0,0.055)) +
  scale_colour_viridis("Text type", discrete = TRUE) +
  scale_x_continuous(limits = c(0, 100), 
                     expand = expansion(add = c(0,1))) +
  labs(title = "Full text vs. Automated") +
  theme_classic() +
  theme(panel.grid = element_blank())

cmb_df = NULL
for(a in unique(abstract_sample_comma$RN)){
  ref_df = subset(abstract_sample_comma, RN == a)
  com_df = subset(taxa_scraped, RN == a)
  if(nrow(com_df) < 1){
    
  } else {
    tally = c()
    for(b in unique(ref_df$species)){
      check = as.numeric(any(com_df$scientific_name == b))
      tally = c(tally, check)
    }
    tmp_df = data.frame(
      RN = a,
      acc = (sum(tally)/length(tally))*100
    )
    cmb_df = rbind(cmb_df, tmp_df)
  }
}
mean(cmb_df$acc)
sd(cmb_df$acc)

den_spec_man_geo = ggplot(data = cmb_df) +
  geom_density(aes(x = acc), size = 1.2) +
  geom_vline(aes(xintercept = mean(acc)), colour = "red") +
  xlab(" ") +
  ylab("Density") +
  scale_y_continuous(expand = c(0, 0), limits = c(0,0.055)) +
  scale_colour_viridis("Text type", discrete = TRUE) +
  scale_x_continuous(limits = c(0, 100), 
                     expand = expansion(add = c(0,1))) +
  labs(title = "Abstract vs. Automated") +
  theme_classic() +
  theme(panel.grid = element_blank())

cmb_df = NULL
for(a in unique(abstract_sample_comma$RN)){
  ref_df = subset(taxa_lpi, RN == a)
  com_df = subset(abstract_sample_comma, RN == a)
  if(nrow(com_df) < 1){
    
  } else {
    tally = c()
    for(b in unique(ref_df$Binomial)){
      check = as.numeric(any(com_df$species == b))
      tally = c(tally, check)
    }
    tmp_df = data.frame(
      RN = a,
      acc = (sum(tally)/length(tally))*100
    )
    cmb_df = rbind(cmb_df, tmp_df)
  }
}
mean(cmb_df$acc)
sd(cmb_df$acc)

den_spec_lpi_man = ggplot(data = cmb_df) +
  geom_density(aes(x = acc), size = 1.2) +
  geom_vline(aes(xintercept = mean(acc)), colour = "red") +
  xlab("Recall (%)") +
  ylab("Density") +
  scale_y_continuous(expand = c(0, 0), limits = c(0,0.055)) +
  scale_colour_viridis("Text type", discrete = TRUE) +
  scale_x_continuous(limits = c(0, 100), 
                     expand = expansion(add = c(0,1))) +
  labs(title = "Full text vs. Abstract") +
  theme_classic() +
  theme(panel.grid = element_blank())


### number of extra species found
# script below working on precision for the automatic scrape - i.e. number of extra species
# manual LPD vs auto
joined_lpi_manual_extra <- full_join(abstract_sample_comma, taxa_scraped, by = c("species" = "scientific_name", "RN")) %>%
  filter(RN %in% abstract_sample$RN) %>%
  group_by(RN) %>%
  mutate(non_na_count = sum(!is.na(taxa_data.class.i.))) %>%
  add_tally() %>%
  select(RN, non_na_count, n) %>%
  unique() %>%
  mutate(prop = (non_na_count/n) * 100) %>%
  mutate(grouping = "Manual abstract/Auto abstract")

### calculating bias
# merge the LPI data with the Catalogue of Life data to pull across the same taxonomies
lpi_main <- left_join(taxa_lpi, unique_col, by = c("Binomial" = "scientific_name")) %>%
  select(RN, taxa_data.class.i., Binomial, taxa_data.order.i.) %>%
  unique() %>%
  filter(!is.na(taxa_data.order.i.)) %>%
  filter(taxa_data.order.i. != "") %>%
  group_by(taxa_data.order.i., taxa_data.class.i.) %>%
  tally() %>%
  ungroup() %>%
  mutate(total = sum(n)) %>%
  mutate(prop = n/total) %>%
  mutate(text_type = "Main text")

# merge the manually searched abstracts with the catalogue of life to bring across taxonomy
lpi_manual <-left_join(abstract_sample_comma, unique_col, by = c("species" = "scientific_name")) %>%
  select(species, taxa_data.order.i., taxa_data.class.i.) %>%
  unique() %>%
  filter(!is.na(taxa_data.order.i.)) %>%
  filter(taxa_data.class.i. != "Bivalvia") %>%
  filter(taxa_data.order.i. != "") %>%
  group_by(taxa_data.order.i., taxa_data.class.i.) %>%
  tally() %>%
  ungroup() %>%
  mutate(total = sum(n)) %>%
  mutate(prop = n/total) %>%
  mutate(text_type = "Abstracts")

# proportion for lpi automatic
lpi_automatic <- taxa_scraped %>% select(RN, taxa_data.class.i., scientific_name, taxa_data.order.i.) %>%
  unique() %>%
  filter(!is.na(taxa_data.order.i.)) %>%
  filter(taxa_data.order.i. != "") %>%
  filter(taxa_data.class.i. != "Malacostraca") %>%
  group_by(taxa_data.order.i., taxa_data.class.i.) %>% 
  tally() %>%
  ungroup() %>%
  mutate(total = sum(n)) %>%
  mutate(prop = n/total) %>%
  mutate(text_type = "Automatic")


# join together the manual and actual and plot correlation
joined_actual <- inner_join(lpi_main, lpi_manual, by = c("taxa_data.order.i.", "taxa_data.class.i."))
joined_auto <- inner_join(lpi_main, lpi_automatic, by = c("taxa_data.order.i.", "taxa_data.class.i."))
joined_abstract <- inner_join(lpi_manual, lpi_automatic, by = c("taxa_data.order.i.", "taxa_data.class.i.")) %>%
  rename("Class" = "taxa_data.class.i.")

# proportion difference plot
joined_auto = subset(joined_auto, n.x > 2)
joined_auto$a <- (joined_auto$prop.y / joined_auto$prop.x) 
joined_abstract = subset(joined_abstract, n.x > 2)
joined_abstract$b <- (joined_abstract$prop.y / joined_abstract$prop.x) 
joined_actual = subset(joined_actual, n.x > 2)
joined_actual$c <- (joined_actual$prop.y / joined_actual$prop.x) 

# phylogenetic tree
taxon_search <- tnrs_match_names(names = joined_auto$taxa_data.order.i., context_name = "All life")

# join the id names back onto the main dataframe
joined_auto$ott_name <- unique_name(taxon_search)
joined_auto$ott_id <- taxon_search$ott_id

# search for trees for those orders
ott_in_tree <- ott_id(taxon_search)[is_in_tree(ott_id(taxon_search))]
tr <- tol_induced_subtree(ott_ids = ott_in_tree)

# dataframe for proportion difference and tips
joined_nodes  <- joined_auto %>%
  select(taxa_data.order.i., taxa_data.class.i., a, ott_name, ott_id) %>%
  mutate(tip_label = paste(ott_name, ott_id, sep = " ott")) %>%
  mutate(tip_label = gsub(" ", "_", tip_label)) %>%
  filter(tip_label %in% tr$tip.label)

# set up node frame
joined_nodes <- data.frame("tip_label" = tr$tip.label, "node" = 1:length(tr$tip.label)) %>%
  right_join(joined_nodes, by = "tip_label")

# change the tip labels
tr$tip.label <- joined_nodes$taxa_data.order.i.

# build the main tree
Tree <- ggtree(tr, layout = "circular") %<+% joined_nodes +
  geom_tiplab(hjust = -0.1, size = 3) +
  theme(text = element_text(size=4), legend.text = element_text(size = 11), legend.title = element_text(size = 12))

# create dataframe of values and tip labels
joined_auto_frame <- joined_auto %>% select(a) %>% mutate(a = log10(a)) %>% data.frame()
joined_abstract_frame <- joined_abstract %>% select(b) %>% mutate(b = log10(b)) %>% data.frame()
joined_actual_frame <- joined_actual %>% select(c) %>% mutate(c = log10(c)) %>% data.frame()

# rename rownames for tip labels
rownames(joined_auto_frame) <- joined_auto$taxa_data.order.i.
rownames(joined_abstract_frame) <- joined_abstract$taxa_data.order.i.
rownames(joined_actual_frame) <- joined_actual$taxa_data.order.i.

# build initial heatmap
p1 <- gheatmap(Tree, joined_auto_frame, offset=13, width=.1,
              colnames_angle=0, colnames_offset_y = 0, color = "white") + 
  scale_fill_gradient2("Bias", midpoint = 0, high = "#D55E00", low = "#0072B2", mid = "white", na.value = "darkgrey", guide = "colourbar",
                       limits = c(-1,1),
                       breaks = c(-1, 0, 1), 
                       labels = c("x0.1", "x1", "x10"))
tmp_vec = joined_auto_frame[,1]
names(tmp_vec) = rownames(joined_auto_frame)
phylosig(compute.brlen(tr), tmp_vec, method = "lambda", test = T)

# add extra head map layer
p2 <- p1 + new_scale_fill()

p2 <- gheatmap(p2, joined_abstract_frame, offset=14.5, width=.1,
               colnames_angle=0, colnames_offset_y = 0, color = "white") + 
  scale_fill_gradient2("Bias", midpoint = 0, high = "#D55E00", low = "#0072B2", mid = "white", na.value = "darkgrey", guide = "colourbar",
                       limits = c(-1,1),
                       breaks = c(-1, 0, 1), 
                       labels = c("x0.1", "x1", "x10"))

tmp_vec = joined_abstract_frame[,1]
names(tmp_vec) = rownames(joined_abstract_frame)
phylosig(compute.brlen(tr), tmp_vec, method = "lambda", test = T)


# add extra head map layer
p3 <- p2 + new_scale_fill()
p3 <- gheatmap(p3, joined_actual_frame, offset=16, width=.1, color = "white") + 
      scale_fill_gradient2("Bias", midpoint = 0, high = "#D55E00", low = "#0072B2", mid = "white", na.value = "darkgrey", guide = "colourbar",
                           limits = c(-1,1),
                           breaks = c(-1, 0, 1), 
                           labels = c("x0.1", "x1", "x10"))+
  theme(
    legend.title=element_text(size=14, hjust = -1),
    legend.text=element_text(size=12),
    legend.position = "right"
  )
tmp_vec = joined_actual_frame[,1]
names(tmp_vec) = rownames(joined_actual_frame)
phylosig(compute.brlen(tr), tmp_vec, method = "lambda", test = T)


    
# remove the tickmarks
g <- ggplotGrob(p3)
g$grobs[[15]][[1]][[1]]$grobs[[5]]$gp$col <- NA


ggarrange(ggarrange(den_spec_lpi_geo, den_spec_man_geo, den_spec_lpi_man, ncol = 1, nrow = 3, labels = c("a)", "b)", "c)")), g, ncol = 2, nrow = 1, labels = c(" ", "      d)"), widths = c(0.4,1))

## Initiate writing to PDF file
ggsave("outputs/taxa_300.pdf", width = 12, height = 8, dpi = 300)

