## scrape abstract species names from LPI abstracts

# vector for packages to install 
packages <- c("dplyr", "taxize", "data.table")

# packages to read in
library(dplyr)
library(taxize)
library(data.table)

# source the functions R script
source("R/00. functions.R")

## read in the LPI texts to scrape 
LPI_abstracts <- read.csv("data/lpi_texts_trends_remove_foreign-text.csv", stringsAsFactors = FALSE)

# run scrape_abs function on Abstract object and time it
system.time({
  all_species <- scrape_abs(data_file = LPI_abstracts,
                            abs_col = "AB",
                            id_col = "RN")
})

# write to csv
write.csv(all_species, "outputs/01. initial_abstract_scrape_LPI_remove_foreign-text.csv")

