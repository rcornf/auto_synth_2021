## script for cleaning the data from the abstract scrape with the COL

# vector for packages to install 
packages <- c("dplyr", "stringr", "data.table")

# packages to read in
library(dplyr)
library(stringr)
library(data.table)

# source the functions R script
source("R/00. functions.R")

## read in the csvs for taxonomic data and the scraped records

# read abstract scrape
all_abs <- read.csv("outputs/01. initial_abstract_scrape_LPI_remove_foreign-text.csv", stringsAsFactors = FALSE) %>%
  mutate(scientific_name = word(scientific_name, 1, 2)) %>%
  mutate(original = scientific_name) %>%
  filter(!is.na(scientific_name))

# read in COL data, change to character and rename variable
unique_col <- readRDS("data/unique_COL_species.rds")
unique_col <- unique_col %>%
  rename(scientific_name = temp) %>%
  mutate_all(as.character) %>%
  mutate(scientific_name = trimws(scientific_name))

# filter out strings with less than one and more than one space
unique_col <- unique_col %>%
  filter(str_count(scientific_name, pattern = " ") == 1)

## below for sequence of merges and cleaning at levels 1, 2, and 3 (direct, punctuation, and " spp")
## merging at level 1a
# merge those that match directly in all_records with the COL data - 7524 unique animal species and 42063 in total
direct_merge <- inner_join(unique_col, all_abs, by = "scientific_name")
direct_merge <- direct_merge %>%
  select(scientific_name, taxa_data.kingdom.i., taxa_data.class.i., taxa_data.order.i., RN, original, taxa_data.scientificNameAuthorship.i., taxa_data.family.i., taxa_data...taxonID.i., taxa_data.acceptedNameUsageID.i., taxa_data.parentNameUsageID.i., taxa_data.taxonomicStatus.i.) %>%
  mutate(level = "1a")

# subset out those merged with the COL data at level 1 
remainder_1 <- left_join(all_abs, unique_col, by = "scientific_name")
remainder_1 <- remainder_1 %>%
  filter(is.na(taxa_data.kingdom.i.)) %>%
  select(scientific_name, RN, original)

## merging at level 1b
# remove any punctuation from string - 157 unique animals and 267 in total
remainder_1$scientific_name <- gsub("([.])|[[:punct:]]", "\\1", remainder_1$scientific_name)
punc_merge <- inner_join(unique_col, remainder_1, by = "scientific_name")
punc_merge <- punc_merge %>%
  mutate(level = "1b")

# subset out those merged with the COL data at level 1b
remainder_2 <- left_join(remainder_1, unique_col, by = "scientific_name")
remainder_2 <- remainder_2 %>%
  filter(is.na(taxa_data.kingdom.i.)) %>%
  select(scientific_name, RN, original)

## merging at level 1c
# remove " spp"
remainder_2$scientific_name <- gsub(" spp", "", remainder_2$scientific_name)
spp_merge <- inner_join(unique_col, remainder_2, by = "scientific_name")
spp_merge <- spp_merge %>%
  mutate(level = "1c")

# subset out those merged with the COL data at level 3
remainder_3 <- left_join(remainder_2, unique_col, by = "scientific_name")
remainder_3 <- remainder_3 %>%
  filter(is.na(taxa_data.kingdom.i.))

## bind all the match records from levels 1, 2, and 3 - 42063 + 267 + 4 = 42334; 7584 unique animals
level_1 <- rbind(direct_merge, punc_merge, spp_merge)
level_1$level <- factor(level_1$level)

## merging at level 2b - abbreviated species
# extract second word of species string in COL
unique_col$second <- unique_col$scientific_name %>% word(2)

# attempt to match first letter and species of abbreviated names to col taxa download
unique_col$ab <- unique_col$scientific_name %>% 
  substr(start = 1, stop = 1) %>%
  paste(".", sep = "") %>%
  paste(unique_col$second, sep = " ")

# remove the second word column
unique_col <- unique_col %>% 
  select(scientific_name, taxa_data.kingdom.i., taxa_data.class.i., taxa_data.order.i., taxa_data.scientificNameAuthorship.i., ab, taxa_data.family.i., taxa_data...taxonID.i., taxa_data.acceptedNameUsageID.i., taxa_data.parentNameUsageID.i., taxa_data.taxonomicStatus.i.)

# change the scientific_name column to ab for merging the abbreviations with the COL
all_abs <- all_abs %>%
  rename(ab = scientific_name)

# merge COL with scraped species - 1611348 potential species matches
abb_merge <- inner_join(all_abs, unique_col, by = "ab")

# convert all to character
abb_merge <- abb_merge %>%
  mutate_all(as.character)

# change ab back to scientific_name in the all_records object
all_abs <- all_abs %>%
  rename(scientific_name = ab)

# remove ab form unique_col
unique_col <- unique_col %>%
  select(-ab)

# create new object of direct_merge for extracting first words
temp_direct <- direct_merge

# extract the first word of each of the resolved scraped species column records
temp_direct$first_word <- temp_direct$scientific_name %>% word(1)

# extract the first word of abb_merge
abb_merge$first_word <- abb_merge$scientific_name %>% word(1)

# run function with file locations from file loc
abbrev_match <- check_abb(locations = abb_merge$RN)

# remove additional columns and add value of 2b for level
abb_merge_genus <- abbrev_match %>%
  mutate(level = "2b") %>%
  filter(!is.na(scientific_name)) %>%
  select(-first_word, - RN) %>%
  rename(RN = loc.i.)

# convert level to factor
abb_merge_genus$level <- factor(abb_merge_genus$level)

# bind the abbreviated matches with the level 1 animal species returned
level_1_2 <- rbind(level_1, abb_merge_genus)

# filter for accepted names
accepted <- level_1_2 %>% 
  filter(taxa_data.taxonomicStatus.i. == "accepted name")

# bind accepted and reslved synonyms
level_1_2 <- accepted%>%
  select(-taxa_data.acceptedNameUsageID.i., -taxa_data...taxonID.i., -taxa_data.scientificNameAuthorship.i., -taxa_data.parentNameUsageID.i., -taxa_data.parentNameUsageID.i., -level) %>%
  unique()

# create genus column
level_1_2$genus <- word(level_1_2$scientific_name, 1, 1)

# identify taxonomic group in common for set of taxa in that abstract
common_taxa <- function(data_file){
  data_subset <- list()
  unique_abstracts <- unique(data_file$RN)
  for(i in 1:length(unique_abstracts)){
    data_subset[[i]] <- data_file %>%
      filter(RN == unique_abstracts[i])
    if(nrow(data_subset[[i]]) == 1){
      data_subset[[i]]$common <- data_subset[[i]]$scientific_name
    }
    else{
      if(length(unique(data_subset[[i]]$genus)) == 1){
        data_subset[[i]]$common <- data_subset[[i]]$genus
      }
      else{
        if(length(unique(data_subset[[i]]$taxa_data.family.i.)) == 1){
          data_subset[[i]]$common <- data_subset[[i]]$taxa_data.family.i.
        }
        else{
          if(length(unique(data_subset[[i]]$taxa_data.order.i)) == 1){
            data_subset[[i]]$common <- data_subset[[i]]$taxa_data.order.i
          }
          else{
            if(length(unique(data_subset[[i]]$taxa_data.class.i.)) == 1){
              data_subset[[i]]$common <- data_subset[[i]]$taxa_data.class.i.
            }
            else{
              if(length(unique(data_subset[[i]]$taxa_data.kingdom.i.)) == 1){
                data_subset[[i]]$common <- data_subset[[i]]$taxa_data.kingdom.i.
                
              }
            }
          }
        }
      }
    }
  }
  data_fin <- rbindlist(data_subset)
  return(data_fin)
}
  
level_1_2_synonym <- common_taxa(level_1_2) %>%
  unique()

# write to csv
write.csv(level_1_2_synonym, "outputs/02. post_COL_species_scrape_LPI_remove_foreign-texts.csv")
