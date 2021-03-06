# import logging to prevent errors
import logging
logging.basicConfig()

# import required modules
from cliff.api import Cliff
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

# read in abstracts
scraped_abstracts = pd.read_csv('C:/Users/joeym/Documents/carnivore_text_mining/data/lpi_texts_trends_remove_foreign-text.csv', encoding = "latin-1")

# assign the localhost address to my_cliff
my_cliff = Cliff('http://localhost:8999')

# result object to append to
result = []

# index for abstract object
abstract  = scraped_abstracts['AB']

# index for title object
EID = scraped_abstracts['RN']

# loop through abstracts
for i in range(0, len(abstract)):

    try:

        # run cliff on text at localhost
        this = my_cliff.parse_text(abstract.iloc[i])
        
        # extract for required part of json for 'mentions'
        this_2 = this['results']
        this_3 = this_2['places']
        this_4 = this_3['mentions']

        # convert json to dataframe
        df = json_normalize(this_4)

        # extract for required part of json for 'focus'
        this_5 = this_3['focus']
        this_6 = this_5['countries']

        # convert focus to dataframe
        this_7 = json_normalize(this_6)
      
        # select the required columns from dataframe
        this_7 = this_7[['name', 'countryCode', 'lat', 'lon']]

        # merge the focal country with the dataframe to add column for focus or not
        df = df.merge(this_7, how = 'outer', on = 'name', indicator = True)

        # add EID to each row of the dataframe
        df['RN'] = EID.iloc[i]

        # rename the merge column as focus level
        df = df.rename(columns = {'_merge':'focus'})

        # select the required columns from dataframe
        df = df[['source.string', 'name', 'countryCode_x', 'lat_x', 'lon_x', 'lat_y', 'lon_y', 'countryCode_y', 'focus', 'RN']]

        # if there are some with right_onlly
        if len(df.loc[df['focus'] == 'right_only']) > 0:

            # remove all x columns and rename y columns, then rename values and remove right_only
            df_2 = df.loc[df['focus'] == 'right_only']
            df_2 = df_2.drop(['lat_x', 'lon_x', 'countryCode_x'], axis = 1)
            df_2 = df_2.rename(columns = {'lat_y':'lat', 'lon_y':'lon', 'countryCode_y':'countryCode'})
            df_2 = df_2.replace('right_only', 'major')
            df = df[df.focus != 'right_only']

            # append df_2 to the result object
            result.append(df_2)

        # select the required columns from dataframe
        df = df[['source.string', 'name', 'countryCode_x', 'lat_x', 'lon_x', 'focus', 'RN']]
        df = df.rename(columns = {'countryCode_x':'countryCode', 'lat_x':'lat', 'lon_x':'lon'})
        df = df.replace('both', 'major')
        df = df.replace('left_only', 'minor')

        # append the df to result
        result.append(df)

        print (i)
 
    # else make a row with just NA
    except IndexError:

        # if nothing found in abstract, make row with empty values
        df = pd.DataFrame({'source.string':[''], 'name':[''],'countryCode':[''], 'lat':[''], 'lon':[''],'focus':[''], 'RN':EID.iloc[i]})

        # append all the dataframes to a list
        result.append(df)

        print ("INDEX_ERROR")

    # else make a row with just NA - if no results found in the geoparse
    except KeyError:

        # if nothing found in abstract, make row with empty values
        df = pd.DataFrame({'source.string':[''], 'name':[''],'countryCode':[''], 'lat':[''], 'lon':[''], 'focus':[''], 'RN':EID.iloc[i]})

        # append all the dataframes to a list
        result.append(df)

        print ("KEY_ERROR")
           
# append the dataframes to result
final = pd.concat(result)

print final

# write to csv

final.to_csv('C:/Users/joeym/Documents/carnivore_text_mining/outputs/03. geoparsed_LPI_remove_foreign-text.csv', sep = ',', encoding = 'utf-8-sig')
