# script for checking the geoparsed locations
# read in the packages
library(rworldmap)
library(rworldxtra)
library(raster)
library(dplyr)
library(ggplot2)
library(here)
library(data.table)
library(forcats)
library(cowplot)
library(viridis)
library(tidyverse)
library(ggraph)
library(tidygraph)
library(patchwork)
library(grid)
library(ggpubr)

# read in the abstract locations, slect for just the lcoation column, and then filter out NA locations
abstract_sample <- read.csv("data/compiled_trend_text.csv") %>%
  select(RN, location) %>%
  group_by(RN) %>%
  tidyr::separate_rows(location, sep = ",")  %>%
  ungroup() %>%
  filter(!location %in% c("Alps", "Western alps", "North atlantic", "Pacific ocean", "Europe")) %>%
  mutate(location = gsub("Scotland", "United Kingdom", location)) %>%
  mutate(location = gsub("Antarctica", "Antarctic", location)) %>%
  mutate(location = gsub("Antarctic", "Antarctica", location)) %>%
  mutate(location = gsub("Tanzania", "United Republic of Tanzania", location)) %>%
  mutate(location = trimws(location)) %>%
  filter(!is.na(location))

# read in the geoparsed location data, filter for only minor mentions, and filter out any continents
geo_locations <- read.csv(("outputs/03. geoparsed_LPI_remove_foreign-text.csv")) %>%
  select(-ï..) %>%
  filter(focus == "major") %>%
  filter(countryCode != "") %>%
  mutate(RN = as.character(RN)) %>%
  select(-source.string) %>%
  select(-countryCode) %>%
  unique() %>%
  filter(!RN %in% c(2587, 2812, 2843))

# split up geo_locations by RN
geo_locations_split <- split(geo_locations, f = geo_locations$RN)

# read in the actual lpi data
lpi_locations <- read.csv(("data/lpi_pops_remove_foreign-text.csv")) %>%
  select(RN, Country, Latitude, Longitude) %>%
  unique() %>%
  mutate(RN = as.character(RN)) %>%
  rename("lon" = "Longitude") %>%
  rename("lat" = "Latitude") %>%
  filter(!RN %in% c(2587, 2812, 2843))

# split up lpi_locations by RN
lpi_locations_split <- split(lpi_locations, f = lpi_locations$RN)

# set geoparsed and lpi as two elements of a list, to be iterated through for extraction of countries
locations_lpi_geo <- list(geo_locations_split, lpi_locations_split)

# extract long/lat columns from geoparsed dataframe and convert to coordinates
coords <- function(geoparsed){
  
  # extract lon/lat columns of geoparsed dataframe 
  study_coords <- data.frame("lon"= geoparsed$lon, 
                             "lat"= geoparsed$lat)
  
  # convert columns to coordinates
  study_coords <- SpatialPointsDataFrame(coords = study_coords[,1:2], 
                                         data = study_coords,
                                         proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"))
  
  # return the final study_coords object
  return(study_coords)
}

# calculate area function
calc_area <- function(map){
  
  # build map
  base_map <- map
  
  # select columns for merging
  area <- base_map@data %>%
    dplyr::select(ADMIN)
  
  # return area object
  return(area)
}

# set up the base map and convert coordinates
get_basemap <- function(){
  
  # download full basemap
  base_map <- getMap(resolution = "high")
  
  # convert to correction projection
  proj4string(base_map) <- CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0")
  
  # return basemap
  return(base_map)
}

# count points within polygons
count_point <- function (map, coordinates){
  
  # find all coordinates that fall within each polygon
  within_base <- over(map, coordinates, returnList = TRUE)
  return(within_base)
}

# function for collapsing the countries of each abstract
collapse_country_lists <- function(data_file){

  # count number of rows for each country name and bind together
  data_file <- lapply(data_file, NROW)
  within_all <- do.call(rbind, data_file)
  
  # turn bound country counts into dataframe, and add rows as a column
  within_frame <- data.frame(within_all)
  within_frame <- setDT(within_frame, keep.rownames = TRUE)[]

  return(within_frame)
  
}

# plot of geographic distribution for lpi and geoparsed
agg_geog <- function(data_file){
  data_fin <- data_file %>%
    select(RN, country) %>%
    unique() %>%
    group_by(country) %>%
    tally() %>%
    ungroup() %>%
    mutate(total = sum(n)) %>%
    mutate(prop = n/total) %>%
    arrange(desc(prop))
  
  return(data_fin)
  
}

# create the basemap from which to check country of points
map <- get_basemap()

# set up empty list for lpi and geoparsed element, and then iterate through both
bound_abstracts_count <- list()
for(j in 1:length(locations_lpi_geo)){

  # identify countries for each abstract, set up new list with each iteration
  geo_locations_split_country <- list()
  for(i in 1:length(locations_lpi_geo[[j]])){
    geo_locations_split_country[[i]] <- count_point(map, coordinates = coords(locations_lpi_geo[[j]][[i]]))

  }

  # assign the names of each abstract on the basis of the names from the original abstract list
  names(geo_locations_split_country) <- names(locations_lpi_geo[[j]])
  

  # collapse the countries of each abstract
  collapsed_countries <- lapply(geo_locations_split_country, collapse_country_lists)
  
  print(collapsed_countries)
  

  # assign the abstract number as a column
  for(i in 1:length(collapsed_countries)){
    collapsed_countries[[i]]$abstract <- names(collapsed_countries)[i]
  }


  
  # bind together the list for each abstract to give the countries in each abstract
  bound_abstracts_count[[j]] <- rbindlist(collapsed_countries) %>%
    filter(within_all != 0) %>%
    rename("country"= "rn") %>%
    rename("count" = "within_all") %>%
    rename("RN" = "abstract") %>%
    select(RN, country, count) %>%
    mutate(RN = as.character(RN))
  
}

bound_abstracts_count[[2]] = bound_abstracts_count[[2]][bound_abstracts_count[[2]]$RN %in% abstract_sample$RN, ]
bound_abstracts_count[[1]] = bound_abstracts_count[[1]][bound_abstracts_count[[1]]$RN %in% abstract_sample$RN, ]

cmb_df = NULL
for(a in unique(bound_abstracts_count[[2]]$RN)){
  ref_df = subset(bound_abstracts_count[[2]], RN == a)
  com_df = subset(bound_abstracts_count[[1]], RN == a)
  tally = c()
  for(b in unique(ref_df$country)){
    check = as.numeric(any(com_df$country == b))
    tally = c(tally, check)
  }
  tmp_df = data.frame(
    RN = a,
    acc = (sum(tally)/length(tally))*100
  )
  cmb_df = rbind(cmb_df, tmp_df)
}

den_loc_lpi_geo = ggplot(data = cmb_df) +
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

mean(cmb_df$acc)
sd(cmb_df$acc)

colnames(abstract_sample)[2] ="country"
abstract_sample$RN = as.character(abstract_sample$RN)
cmb_df = NULL
for(a in unique(bound_abstracts_count[[2]]$RN)){
  ref_df = subset(bound_abstracts_count[[2]], RN == a)
  com_df = subset(abstract_sample, RN == a)
  if(nrow(com_df) < 1){
    
  } else {
    tally = c()
    for(b in unique(ref_df$country)){
      check = as.numeric(any(com_df$country == b))
      tally = c(tally, check)
    }
    tmp_df = data.frame(
      RN = a,
      acc = (sum(tally)/length(tally))*100
    )
    cmb_df = rbind(cmb_df, tmp_df)
  }
}

den_loc_lpi_man = ggplot(data = cmb_df) +
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

mean(cmb_df$acc)
sd(cmb_df$acc)


cmb_df = NULL
for(a in unique(abstract_sample$RN)){
  ref_df = subset(abstract_sample, RN == a)
  com_df = subset(bound_abstracts_count[[1]], RN == a)
  if(nrow(com_df) < 1){
    
  } else {
    tally = c()
    for(b in unique(ref_df$country)){
      check = as.numeric(any(com_df$country == b))
      tally = c(tally, check)
    }
    tmp_df = data.frame(
      RN = a,
      acc = (sum(tally)/length(tally))*100
    )
    cmb_df = rbind(cmb_df, tmp_df)
  }
}

den_loc_man_geo = ggplot(data = cmb_df) +
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

mean(cmb_df$acc)
sd(cmb_df$acc)


### map for difference between proportion for geoparsed and lpi
# first recalculate the proportions for both the lpi and geoparsed data
geo_prop <- agg_geog(bound_abstracts_count[[1]])
lpi_prop <- agg_geog(bound_abstracts_count[[2]])
manual_prop <- agg_geog(abstract_sample)

# join together the lpi and geoparsed data to calculate the proportion difference
proportion_diff_lpi_geo <- left_join(lpi_prop, geo_prop,  by = "country") %>%
  mutate(prop.y = ifelse(is.na(prop.y), 0, prop.y)) %>%
  mutate(prop_diff = prop.y - prop.x)

proportion_diff_man_geo <- left_join(manual_prop, geo_prop,  by = "country") %>%
  mutate(prop.y = ifelse(is.na(prop.y), 0, prop.y)) %>%
  mutate(prop_diff = prop.y - prop.x)

proportion_diff_lpi_man <- left_join(lpi_prop, manual_prop,  by = "country") %>%
  mutate(prop.y = ifelse(is.na(prop.y), 0, prop.y)) %>%
  mutate(prop_diff = prop.y - prop.x)

# proportion difference plot
proportion_diff_lpi_geo$diff <- log10((proportion_diff_lpi_geo$prop.y / proportion_diff_lpi_geo$prop.x))
proportion_diff_lpi_geo$diff = ifelse(is.infinite(proportion_diff_lpi_geo$diff), -1, proportion_diff_lpi_geo$diff)
proportion_diff_lpi_geo = subset(proportion_diff_lpi_geo, n.x > 2)

proportion_diff_man_geo$diff <- log10((proportion_diff_man_geo$prop.y / proportion_diff_man_geo$prop.x))
proportion_diff_man_geo$diff = ifelse(is.infinite(proportion_diff_man_geo$diff), -1, proportion_diff_man_geo$diff)
proportion_diff_man_geo = subset(proportion_diff_man_geo, n.x > 2)

proportion_diff_lpi_man$diff <- log10((proportion_diff_lpi_man$prop.y / proportion_diff_lpi_man$prop.x))
proportion_diff_lpi_man$diff = ifelse(is.infinite(proportion_diff_lpi_man$diff), -1, proportion_diff_lpi_man$diff)
proportion_diff_lpi_man = subset(proportion_diff_lpi_man, n.x > 2)

# create a fortified basemap for ggplot
map <- get_basemap()
map_fort <- fortify(map)

# underneath basemap
map_base <- get_basemap()
map_base_fort <- fortify(map_base)

# merge polygon area, within point count, and proportion count; and join records to main map
area_within <- inner_join(proportion_diff_lpi_geo, calc_area(map = get_basemap()), by = c("country" = "ADMIN"))
within_map <- inner_join(area_within, map_fort, by = c("country" = "id")) %>%
  mutate(grouping = "ab_auto")
  
# build map for proportion difference as a factor
map_lpi_geo = ggplot() + 
  geom_polygon(aes(x = long, y = lat, group = group), fill = "white", data = map_base_fort, colour = "grey40", size = 0.1) +
  geom_polygon(aes(x = long, y = lat, group = group, fill = diff), data = within_map) +
  scale_fill_gradient2("Bias", midpoint = 0, 
                       high =  "#D55E00", #"firebrick2", #
                       low = "#0072B2", 
                       mid = "gainsboro",  #"#F6BB42", #"white", 
                       na.value = "white", 
                       guide = "colourbar",
                       limits = c(-0.69897,0.69897),
                       breaks = c(-0.69897, 0, 0.69897),
                       labels = c("x0.2", "x1", "x5")
                       ) + 
  coord_map(projection = "mollweide", xlim = c(-180,180)) +
  theme(axis.text = element_blank(), 
        axis.ticks = element_blank(), 
        axis.title = element_blank(),
        axis.line = element_blank(),
        text = element_text(size = 14),
        panel.background = element_rect(fill = "white"),
        plot.margin = unit(c(0,0,1,0), "cm"), 
        legend.position = "none")

# merge polygon area, within point count, and proportion count; and join records to main map
area_within <- inner_join(proportion_diff_man_geo, calc_area(map = get_basemap()), by = c("country" = "ADMIN"))
within_map <- inner_join(area_within, map_fort, by = c("country" = "id")) %>%
  mutate(grouping = "ab_auto")

# build map for proportion difference as a factor
map_man_geo = ggplot() + 
  geom_polygon(aes(x = long, y = lat, group = group), fill = "white", data = map_base_fort, colour = "grey40", size = 0.1) +
  geom_polygon(aes(x = long, y = lat, group = group, fill = diff), data = within_map) +
  scale_fill_gradient2("Bias", midpoint = 0, 
                       high =  "#D55E00", #"firebrick2", #
                       low = "#0072B2", 
                       mid = "gainsboro",  #"#F6BB42", #"white", 
                       na.value = "white", 
                       guide = "colourbar",
                       limits = c(-0.69897,0.69897),
                       breaks = c(-0.69897, 0, 0.69897),
                       labels = c("x0.2", "x1", "x5")
                       ) + 
  coord_map(projection = "mollweide", xlim = c(-180,180)) +
  theme(axis.text = element_blank(), 
        axis.ticks = element_blank(), 
        axis.title = element_blank(),
        axis.line = element_blank(),
        text = element_text(size = 14),
        panel.background = element_rect(fill = "white"),
        plot.margin = unit(c(0,0,1,0), "cm"), 
        legend.position = "none")

# merge polygon area, within point count, and proportion count; and join records to main map
area_within <- inner_join(proportion_diff_lpi_man, calc_area(map = get_basemap()), by = c("country" = "ADMIN"))
within_map <- inner_join(area_within, map_fort, by = c("country" = "id")) %>%
  mutate(grouping = "ab_auto")

# build map for proportion difference as a factor
map_lpi_man = ggplot() + 
  geom_polygon(aes(x = long, y = lat, group = group), fill = "white", data = map_base_fort, colour = "grey40", size = 0.1) +
  geom_polygon(aes(x = long, y = lat, group = group, fill = diff), data = within_map) +
  scale_fill_gradient2("Bias", midpoint = 0, 
                       high =  "#D55E00", #"firebrick2", #
                       low = "#0072B2", 
                       mid = "gainsboro",  #"#F6BB42", #"white", 
                       na.value = "white", 
                       guide = "colourbar",
                       limits = c(-0.69897,0.69897),
                       breaks = c(-0.69897, 0, 0.69897),
                       labels = c("x0.2", "x1", "x5")
                       ) + 
  coord_map(projection = "mollweide", xlim = c(-180,180)) +
  theme(axis.text = element_blank(), 
        axis.ticks = element_blank(), 
        axis.title = element_blank(),
        axis.line = element_blank(),
        text = element_text(size = 14),
        panel.background = element_rect(fill = "white"),
        plot.margin = unit(c(0,0,1,0), "cm"), 
        legend.position = "bottom")

# remove the tickmarks


## bipartite plots for United States
# main text joined onto automated

# change the name of the location column to country to fit with the network function
abstract_manual_locations <- abstract_sample %>%
  mutate(RN = as.character(RN)) %>%
  unique() 
  
# abstract numbers for the manual abstracts
abstract_IDs <- read.csv("data/compiled_trend_text.csv") %>%
  pull(RN) %>% 
  unique() 

# main text locations
main_text_locations <- bound_abstracts_count[[2]] %>%
  select(RN, country) %>%
  unique() %>%
  filter(RN %in% abstract_IDs)

# geoparsed locations
geoparsed_locations <- bound_abstracts_count[[1]] %>%
  select(RN, country) %>%
  unique() %>%
  filter(RN %in% abstract_IDs)

# function to create data for either manual abstracts and main text or geoparsed abstracts and main text
create_network_data <- function(comparison, reference, country){
  # abstract manual locations joined onto the main text
  geo_abs_sample <- full_join(comparison, reference, by = "RN")
  
  # subset the USA abstracts from the joined abstract data
  geo_abs_sample_US <- geo_abs_sample %>%
    #filter(RN %in% USA_abs[!is.na(USA_abs)]) %>% 
    filter(country.y == country) %>% 
    select(country.x, RN, country.y) %>%
    filter(!is.na(country.x)) %>%
    filter(!is.na(country.y)) %>%
    mutate(country.x = paste(country.x, "geo", sep = "-")) %>%
    mutate(country.y = paste(country.y, "lpi", sep = "-"))
  
  # set up the source and destination of the bipartite plot
  sources <- geo_abs_sample_US %>%
    distinct(country.x) %>%
    rename(label = country.x)
  
  destinations <- geo_abs_sample_US %>%
    distinct(country.y) %>%
    rename(label = country.y)
  
  # set up the nodes
  nodes <- full_join(sources, destinations, by = "label")
  
  # create an id column and remove geo/lpi for plot
  nodes <- nodes %>% rowid_to_column("id")
  
  # weight the edges
  per_route <- geo_abs_sample_US %>%  
    group_by(country.x, country.y) %>%
    summarise(weight = n()) %>% 
    ungroup()
  
  # create from and to columns
  edges <- per_route %>% 
    left_join(nodes, by = c("country.x" = "label")) %>% 
    rename(from = id)
  
  edges <- edges %>% 
    left_join(nodes, by = c("country.y" = "label")) %>% 
    rename(to = id)
  
  # to create columns for match, need to remove geo and lpi from strings first
  edges <- edges %>%
    mutate(country.x = gsub("-geo", "", country.x)) %>%
    mutate(country.y = gsub("-lpi", "", country.y))
  
  # create column for match
  edges$matched[edges$country.x == edges$country.y] <- "match"
  edges$matched[edges$country.x != edges$country.y] <- "no-match"
  
  # reorder the columns, including whether it matches
  edges <- select(edges, from, to, weight, matched)
  
  # change labels for geo/lpi for plot
  nodes <- nodes %>%
    mutate(label = gsub("-geo", "", label)) %>%
    mutate(label = gsub("-lpi", "", label))
  
  # convert to a tidy format for the network plot
  routes_tidy <- tbl_graph(nodes = nodes, edges = edges, directed = TRUE)
  

  return(routes_tidy)
  
}

# set up the network data for the automated abstract data
routes_tidy_geo <- create_network_data(geoparsed_locations, main_text_locations, "France")

# extract the layout
layout_geo <- create_layout(routes_tidy_geo, layout = "sugiyama")

layout_geo$x = c(7,6,5,4,3,2,1,7)
layout_geo$label[layout_geo$label == "United States of America"] <- "USA"

# set up data labels
heading_labels_auto <- data.frame(x = c(8, 8), y = c(1, 2), labels = c("Full text", "Automated"))

# plot the network
net_lpi_geo_FRA <- ggraph(layout_geo) +  
  #geom_node_point(alpha = 0.7) + 
  theme_graph() + 
  geom_label(aes(x = x, y = y, label = labels), data = heading_labels_auto, size = 3, fontface = "bold", fill = "gray93") +
  coord_flip(xlim = c(0, 11), ylim = c(0.7, 2.3)) +
  scale_edge_color_manual("Match", values = c("black", "red")) +
  geom_edge_link(aes(width = weight, colour = matched), alpha = 0.3) +
  geom_node_text(aes(label = label), size = 3) +
  expand_limits(y = 2) +
  scale_alpha_manual(guide = "none", values = c(rep(0.85, 8), 1)) + 
  theme(plot.margin = unit(c(1,1,1,1), "cm")) +
  theme(legend.position = "none")
  

# set up the network data for the automated abstract data
routes_tidy_geo <- create_network_data(geoparsed_locations, abstract_manual_locations, "France")

# extract the layout
layout_geo <- create_layout(routes_tidy_geo, layout = "sugiyama")

layout_geo$x = c(9,8,7,6,5,4,3,2,1,9)
layout_geo$label[layout_geo$label == "United States of America"] <- "USA"

# set up data labels
heading_labels_auto <- data.frame(x = c(10, 10), y = c(1, 2), labels = c("Abstract", "Automated"))

# plot the network
net_man_geo_FRA <- ggraph(layout_geo) +  
  #geom_node_point(alpha = 0.7) + 
  theme_graph() + 
  geom_label(aes(x = x, y = y, label = labels), data = heading_labels_auto, size = 3, fontface = "bold", fill = "gray93") +
  coord_flip(xlim = c(0, 11), ylim = c(0.7, 2.3)) +
  scale_edge_color_manual("Match", values = c("black", "red")) +
  geom_edge_link(aes(width = weight, colour = matched), alpha = 0.3) +
  geom_node_text(aes(label = label), size = 3) +
  expand_limits(y = 2) +
  scale_alpha_manual(guide = "none", values = c(rep(0.85, 8), 1)) + 
  theme(plot.margin = unit(c(1,1,1,1), "cm")) +
  theme(legend.position = "none")


# set up the network data for the automated abstract data
routes_tidy_geo <- create_network_data(abstract_manual_locations, main_text_locations, "France")

# extract the layout
layout_geo <- create_layout(routes_tidy_geo, layout = "sugiyama")

layout_geo$x = c(3,2,3)
layout_geo$label[layout_geo$label == "United States of America"] <- "USA"

# set up data labels
heading_labels_auto <- data.frame(x = c(4,4), y = c(1, 2), labels = c("Full text", "Abstract"))

# plot the network
net_lpi_man_FRA <- ggraph(layout_geo) +  
  #geom_node_point(alpha = 0.7) + 
  theme_graph() + 
  geom_label(aes(x = x, y = y, label = labels), data = heading_labels_auto, size = 3, fontface = "bold", fill = "gray93") +
  coord_flip(xlim = c(0,11),ylim = c(0.7, 2.3)) +
  scale_edge_color_manual("Match", values = c("black", "red")) +
  geom_edge_link(aes(width = weight, colour = matched), alpha = 0.3) +
  geom_node_text(aes(label = label), size = 3) +
  expand_limits(y = 2) +
  scale_alpha_manual(guide = "none", values = c(rep(0.85, 8), 1)) + 
  theme(plot.margin = unit(c(1,1,1,1), "cm")) +
  theme(legend.position = "none")

ggarrange(den_loc_lpi_geo, map_lpi_geo, net_lpi_geo_FRA, 
          den_loc_man_geo, map_man_geo, net_man_geo_FRA, 
          den_loc_lpi_man, map_lpi_man, net_lpi_man_FRA, 
          ncol = 3, nrow = 3,
          common.legend = T, legend = "bottom", widths = c(0.4,1,0.4), labels = c(
            "1a)", "            1b)","1c)", "2a)", "            2b)", "2c)", "3a)", "            3b)", "3c)"
          ))
ggsave("outputs/geography_300.pdf", width = 12, height = 10, dpi = 300)

