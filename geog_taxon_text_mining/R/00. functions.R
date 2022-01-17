##  script for all functions used as part of the text analysis

## scrape_abs function
# scrape_abs() will scrape each abstract, 
# assign NA where no species are returned, 
# merge with DOI, 
# incrementally combine outputs, 
# and then print all as all_abstracts.
# Takes one argument, in form of Abstract object created above
scrape_abs <- function(data_file, abs_col, id_col){
  
  # create empty list for combining species lists
  datalist <- list()  
  
  # iterate scrapenames Taxize function across each abstract - 
  # will need to be changed to iterate across length of Abstract object
  for (i in 1:nrow(data_file)) {
    
    # run tryCatch to catch any error abstracts, and print the DOI at that count
    tryCatch({
      
      #browser()

      # run scrapenames on each value in Abstract column
      species <- scrapenames(text = data_file[[abs_col]][i])
      
      #print(species$data)
      
      
      # if species are found, add the names and DOI to a temp dataframe, rename columns, and then add to list
      if (length(species$data > 0)){
        temp <- data.frame(species$data$scientificname, data_file[[id_col]][i])
        colnames(temp) <- c("scientific_name", id_col)
        datalist[[i]] <- temp
      }
      
      # if species are not found, create dataframe with NA and the DOI for that row, rename columns, and add to list
      else {
        temp_2 <- data.frame(NA, data_file[[id_col]][i])
        colnames(temp_2) <- c("scientific_name", id_col)
        datalist[[i]] <- temp_2
      }
      
      # print the DOI for the error at that abstract count
    }, error = function(x) print(c(i, data_file[[id_col]][i])))
  }
  
  # combine the results of each datalist iteration and then print
  all_abstracts <- rbindlist(datalist)
  return(all_abstracts)
}

### function for removing author column from the scientific name column
species <- function(taxa_data, count){
  
  # create a list
  data <- list()  
  
  # iterate over the number of counts defined as an argument
  for (i in 1:count){
    
    # catch errors
    tryCatch({
      
      # whenever see the pattern of author in scientific name, remove it, make a dataframe from rows at that iteration, save to list
      temp <- gsub(taxa_data$scientificNameAuthorship[i], "", taxa_data$scientificName[i])
      temp_spec <- data.frame(temp, taxa_data$kingdom[i], taxa_data$class[i], taxa_data$order[i], taxa_data$scientificNameAuthorship[i], taxa_data$family[i], taxa_data$..taxonID[i], taxa_data$acceptedNameUsageID[i], taxa_data$parentNameUsageID[i], taxa_data$taxonomicStatus[i])
      data[[i]] <- temp_spec
      
      # print iteration number when error encountered
    }, error = function(x) print(c(i, taxa_data$scientificName[i])))
  }
  
  # bind the data lists and return the final bound object 
  species_names <- rbindlist(data)
  return(species_names)
}

# remerge only with those that appear in each paper
# find all the direct merges at each DOI
check_abb <- function(locations){
  
  # set up empty list objects
  loc <- unique(locations)
  abb <- list()
  direct <- list()
  joined <- list()
  new_joined <- list()
  
  # loop through list of DOIs in abb_merge, remove scientific names, and assign to abb as i element
  for (i in 1:length(loc)){
    abb_spec <- abb_merge %>% 
      filter(RN == loc[i]) %>%
      filter(!duplicated(scientific_name)) %>%
      select(scientific_name, first_word, taxa_data.kingdom.i., taxa_data.class.i., taxa_data.order.i., original, taxa_data.scientificNameAuthorship.i., RN, taxa_data.family.i., taxa_data...taxonID.i., taxa_data.acceptedNameUsageID.i., taxa_data.parentNameUsageID.i., taxa_data.taxonomicStatus.i.)
    abb[[i]] <- abb_spec
    
    # loop through list of DOIs in temp_direct, remove scientific names, and assign to abb as i element
    direct_spec <- temp_direct %>% 
      filter(RN == loc[i]) %>%
      filter(!duplicated(scientific_name)) %>%
      select(first_word)
    direct[[i]] <- direct_spec
    
    # if there are mentions in both the direct_spec and abb_spec join by first word
    if (length(direct_spec$first_word) > 0 & length(abb_spec$scientific_name) > 0) {
      
      # join abb and direct by first word
      joined <- inner_join(abb[[i]], direct[[i]], by = "first_word")
      
      # there are matches return build into dataframe
      if (length(joined$scientific_name > 0)) {
        temp_1 <- data.frame(joined, loc[i])
        new_joined[[i]] <- temp_1
      }
      
      # otherwise add NA at that row
      else {
        joined[1,] <- NA
        temp_2 <- data.frame(joined, loc[i])
        new_joined[[i]] <- temp_2
      }
    }
    
    # if there are no direct_spec or abb_spec in that DOI return row of NAs
    else {
      joined <- data.frame(scientific_name = NA, first_word = NA, taxa_data.kingdom.i. = NA, taxa_data.class.i. = NA, taxa_data.order.i. = NA, original = NA, taxa_data.scientificNameAuthorship.i. = NA, RN = NA, taxa_data.family.i. = NA, taxa_data...taxonID.i. = NA, taxa_data.acceptedNameUsageID.i. = NA, taxa_data.parentNameUsageID.i. = NA, taxa_data.taxonomicStatus.i. = NA)
      temp_3 <- data.frame(joined, loc[i])
      new_joined[[i]] <- temp_3
    }
  }
  
  # bind all lists and return
  fin <- rbindlist(new_joined)
  return(fin)
}

# remove string after copyright sign
remove_after_copyright <- function(data_file){
  
  data <- list()
  
  # iterate through each of the downloads abstract object
  for(i in 1:nrow(data_file)) ({
    
    # if copyright symbol in the abstract, assign a boolean
    logical_copy <- grepl("textcopyright", data_file$ABSTRACT[i])
    
    # remove characters after the copyright symbol if it's there
    if(logical_copy == TRUE) ({
      
      # remove any characters after the copyright symbol
      data_file$ABSTRACT[i] <- gsub("textcopyright.*","", data_file$ABSTRACT[i])
    })
    
    data[[i]] <- data.frame("ABSTRACT" = data_file$ABSTRACT[i], "RN" = data_file$RN[i])
  })
  
  cleaned <- rbindlist(data)
  return(cleaned)
}

# format geoparsed data
form_geoparse <- function(data, foc, continents, oddities, code_out){
  form <- data %>%
    dplyr::select(-confidence) %>%
    dplyr::filter(!is.na(lat)) %>%
    dplyr::filter(!name %in% continents) %>%
    dplyr::filter(!name %in% oddities) %>%
    dplyr::filter(focus %in% foc) %>%
    dplyr::filter(!grepl(code_out, countryCode))
  
  return(form)
  
}

speciesify <- function(scraped, first_word, last_word) {
  
  species <- scraped
  
  # onle keep 1st and 2nd words in species column
  species$scientific_name <- species$scientific_name %>% word(first_word, last_word)
  
  # return scraped
  return(species)
}