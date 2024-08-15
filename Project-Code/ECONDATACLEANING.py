#Eliot Ozaki
#ECON DATA CLEANING for Income effect on Emissions Research Project


### Part 1: Data Cleaning/Merging of Population and Emissions Datasets

## Imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap
from pandas_profiling import ProfileReport

## Load the datasets
population_df = pd.read_csv("C:/Users\eliot/OneDrive/Desktop/ECONPROJECT/Cleaned-Datasets/POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Base-Datasets/COUNTY_EMISSIONS2021.csv")

##Create a dictionary of state names and their initials
state_initials = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
print(state_initials['California'])  # Output: CA



## Using the dictionary
states = list(state_initials.keys())
initials = list(state_initials.values())
population_df.tail(20)

# Changing state names to initials
for state in states:
    population_df.loc[population_df['State']==state,'State'] = state_initials[state]
    
print(population_df['State'].unique())
print(population_df['County'][population_df['County'].apply(len) <= 2])

# Print the number of unique counties per state in the population dataset
index = 0
popcounts = 0
for initial in initials:
    count = (population_df['State'] == initial).sum()
    print(f"{initial}: {count}")

for initial in initials:
    count = (emissions_df['State'] == initial).sum()
    print(f"{initial}: {count}")


## Findind differences between the datasets, and cleaning them up

# Import both as new dataframes
population_df = pd.read_csv("C:/Users\eliot/OneDrive/Desktop/ECONPROJECT/Cleaned-Datasets/POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Base-Datasets/COUNTY_EMISSIONS2021.csv")

# Group by 'State' and count the number of unique counties
emissions_counties = emissions_df.groupby('State')['County'].nunique().reset_index(name='Emissions_Counties')
population_counties = population_df.groupby('State')['County'].nunique().reset_index(name='Population_Counties')

# Merge the results on 'State'
merged_df = pd.merge(emissions_counties, population_counties, on='State', how='outer')

# Fill NaN values with 0 (if any state is missing in one of the datasets)
merged_df.fillna(0, inplace=True)

# Convert to integers (since counts should be integers)
merged_df['Emissions_Counties'] = merged_df['Emissions_Counties'].astype(int)
merged_df['Population_Counties'] = merged_df['Population_Counties'].astype(int)

# Compare the results
merged_df['Difference'] = merged_df['Emissions_Counties'] - merged_df['Population_Counties']

# Display the result
print(merged_df)
print(merged_df[merged_df['Difference']!=0])
print(merged_df['Emissions_Counties'][merged_df['State']=='DE'])
print(emissions_df['County'][emissions_df['State']=='DE'])

# Find the states with differences
states_with_differences = merged_df['State'][merged_df['Difference'] != 0].tolist()

# Printing the states with differences and what counties they differe in
for state in states_with_differences:
    pop_counties = set(population_df[population_df['State'] == state]['County'])
    emis_counties = set(emissions_df[emissions_df['State'] == state]['County'])
    diff_counties = pop_counties - emis_counties
    
    if diff_counties:
        print(f"State: {state}")
        print("Counties in Population dataset but not in Emissions dataset:")
        for county in diff_counties:
            print(county)
        print()

print(len(population_df))
print(len(emissions_df))

# Setting index to 'County' for both datasets
emissions_df.set_index("County")
population_df.set_index("County")

# Merging the datasets
df = pd.merge(population_df,emissions_df)

df.describe()
df.columns

# Exporting the cleaned datasets
population_df.to_csv("POPULATIONDATA-Cleaned.csv")
emissions_df.to_csv("EMISSIONSDATA-Cleaned.csv")
df.to_csv("AllData-MergedDS.csv")



### Part 2: Data Cleaning/Merging of GDP Dataset

#Cleaning More Datasets
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap
#from pandas_profiling import ProfileReport

## Cleaning
# Load the datasets
population_df = pd.read_csv("Cleaned-Datasets\POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("Cleaned-Datasets/EMISSIONSDATA-Cleaned.csv")
merged_df = pd.read_csv("AllData-MergedDS.csv")

# Reit. dictionary of state names and their initials
state_initials = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
print(state_initials['California'])  # Output: CA

# More reit.
states = list(state_initials.keys())
initials = list(state_initials.values())

# Trying to load gdp dataset
import chardet
with open("Cleaned-Datasets\County-MSA-GDP-DATA.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

# Load the GDP dataset
gdp_df = pd.read_csv("Cleaned-Datasets\County-MSA-GDP-DATA.csv",encoding="Windows-1252")

# Initial look at the dataset
gdp_df.describe()
gdp_df.head(10)

## Inital Cleaning

# Remove rows with Geoname of United States and any aggregated state rows.
gdp_df = gdp_df[~gdp_df['GeoName'].isin(states + ['United States'])]

# Check gdp df length
print(len(gdp_df)/3)

# Check differences from gdp and merged datasets
counties_not_in_gdp = set(merged_df['County'].str.split(',').str[0]) - set(gdp_df['GeoName'].str.split(',').str[0])
print("Counties in merged_df but not in gdp_df:")
for county in counties_not_in_gdp:
    print(county)
counties_not_in_merged = set(gdp_df['GeoName'].str.split(',').str[0]) - set(merged_df['County'].str.split(',').str[0])
print("Counties in gdp but not in merged:")
for county in counties_not_in_merged:
    print(county)

## Cleaning the GDP dataset
gdp_df.loc[gdp_df['GeoName'].str.contains(r' \(Independent City\)', regex=True), 'GeoName'] = \
    gdp_df['GeoName'].str.replace(r' \(Independent City\)', '', regex=True).str.strip().apply(lambda x: "City of " + x)
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('St.', 'St')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('East', 'E')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('Census Area', '')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('City and Borough', '')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('Borough', '')
gdp_df.loc[gdp_df['GeoName'] =='LaSalle', 'GeoName'] = 'La Salle'

#Doing some checks on specific values
print(gdp_df.loc[gdp_df['GeoName'].str.contains('Emporia'),'GeoName'])
print(merged_df.loc[merged_df['County'].str.contains('Columbia'),'County'])
print(gdp_df.loc[~gdp_df['GeoName'].str.contains(','),'GeoName'])
print(merged_df.loc[merged_df['County'].str.contains('Rocky Mountain'),'County'])

#Setting Index of gdp df
gdp_df.set_index("GeoName")



## Pivoting the GDP dataset

pivoted_df = gdp_df.pivot_table(index=['GeoName', 'GeoFIPS'], columns='Unit', values='GDP(Thousands)', aggfunc='first')

# Flatten the column MultiIndex, if necessary
pivoted_df.columns = [col for col in pivoted_df.columns]
# Reset the index to turn GeoName back into a column
pivoted_df.reset_index(inplace=True)
print(pivoted_df)


# Separate GeoName into County and State columns
pivoted_df[['County', 'State']] = pivoted_df['GeoName'].str.split(', ', expand=True).iloc[:, [0, 1]]
pivoted_df['County'] = pivoted_df['County'].str.rstrip()
pivoted_df.head(10)
pivoted_df.shape
pivoted_df.set_index("County")
pivoted_df.sort_values(["State", "County"], inplace=True)
merged_df.sort_values(["State", "County"], inplace=True)
merged_df.columns

# looking at merged and pivoted dfs.
merged_df[['County','State']].head(10)
pivoted_df[['County','State']].head(10)

## Finding differences between pivoted and merged dataset
pivoted_counties = pivoted_df.groupby('State')['County'].nunique().reset_index(name='Pivoted_Counties')
merged_counties = merged_df.groupby('State')['County'].nunique().reset_index(name='Merged_Counties')

# Merge the results on 'State'
merged_counts_df = pd.merge(pivoted_counties, merged_counties, on='State', how='outer')

# Fill NaN values with 0 (if any state is missing in one of the datasets)
merged_counts_df.fillna(0, inplace=True)

# Convert to integers (since counts should be integers)
merged_counts_df['Pivoted_Counties'] = merged_counts_df['Pivoted_Counties'].astype(int)
merged_counts_df['Merged_Counties'] = merged_counts_df['Merged_Counties'].astype(int)

# Compare the results
merged_counts_df['Difference'] = merged_counts_df['Pivoted_Counties'] - merged_counts_df['Merged_Counties']

# Display the result
print(merged_counts_df)
print(merged_counts_df[merged_counts_df['Difference'] != 0])
print(merged_counts_df['Pivoted_Counties'][merged_counts_df['State'] == 'DE'])
print(merged_df['County'][merged_df['State'] == 'DE'])

# Printing states with differences in number of counties
states_with_differences = merged_counts_df['State'][merged_counts_df['Difference'] != 0].tolist()
print(states_with_differences)

## Removing NA rows and doing more specific cleaning
na_rows = pivoted_df[pivoted_df['State'].isna()]
print(na_rows)
nostate_counties = na_rows['County'].unique()
print(merged_df[merged_df['County'].isin(nostate_counties)][['County', 'State']])
pivoted_df.loc[pivoted_df['County']=="District of Columbia", 'State'] = 'DC'
pivoted_df = pivoted_df.dropna(subset=['State'])
pivoted_df.loc[pivoted_df['County'].str.startswith('West '), 'County'] = pivoted_df['County'].str.replace('West', 'W')

# Removing the spaces from the end of counties who end in spaces.
for county in merged_df['County']:
    if county.endswith(' '):
        print(county)
        merged_df['County'] = merged_df['County'].str.rstrip()

#######CANNOT RUN YET, NEED TO ELIM NAS IN PIVOTED_DF
# Printing rows with NA values in pivoted_df
pivoted_df = pivoted_df.dropna()
#counties_with_asterisk = pivoted_df[pivoted_df['State'].str.contains('\*')][['County', 'State']]
#print(counties_with_asterisk)

## Checking what counties are in one df but not the other
# ** IMPORTANT 
counties_not_in_gdp = set(merged_df['County']) - set(pivoted_df['County'])
counties_not_in_gdp = sorted(counties_not_in_gdp)
print("Counties in merged_df but not in gdp_df:")
for county in counties_not_in_gdp:
    print(county)

counties_not_in_merged = set(pivoted_df['County']) - set(merged_df['County'])
counties_not_in_merged = sorted(counties_not_in_merged)
print("Counties in gdp but not in merged (sorted alphabetically):")
for county in counties_not_in_merged:
    print(county)


## Finding rows in pivoted_df with asterisks in the 'State' column
print(pivoted_df[pivoted_df['State'].str.contains('\*')])
edited_counties = pivoted_df[pivoted_df['State'].str.contains('\*')]
edited_counties.head(len(edited_counties))

# Using that list, finding counties that are in the merged_df/not in merged df
counties_not_in_merged = []
for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        counties_not_in_merged.append(county)
print(counties_not_in_merged)

counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(counties_in_merged)

# Removing the asterix from the 'State' column from the rows with counties in merged_df
pivoted_df.loc[pivoted_df['County'].isin(counties_in_merged), 'State'] = pivoted_df['State'].str[:2]
print(pivoted_df[pivoted_df['State'].str.contains('\*')])


## Cleaning up the counties that have asterixes in the 'State' column without plusses.
counties_without_plus = [county for county in counties_not_in_merged if '+' not in county]
print(counties_without_plus)

from difflib import get_close_matches
similar_counties = {}

for county in counties_not_in_merged:
    matches = get_close_matches(county, merged_df['County'],n=1,cutoff=0.7)
    if matches:
        similar_counties[county] = matches[0]

print(similar_counties)

## More specific cleaning based on those counties
print(pivoted_df[pivoted_df['County'] == 'Skagway-Hoonah-Angoon'])
# Dropping the row with 'Skagway-Hoonah-Angoon' in the 'County' column from gdp df (pivoted)
pivoted_df = pivoted_df[pivoted_df['County'] != 'Skagway-Hoonah-Angoon']
edited_counties = edited_counties[edited_counties['County'] != 'Skagway-Hoonah-Angoon']

## Re-running prints of different combinations of counties with/without asterisks and plusses (After removing some counties)
counties_not_in_merged = []
for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        counties_not_in_merged.append(county)
print(counties_not_in_merged)
counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(counties_in_merged)

pivoted_df.loc[pivoted_df['County'].isin(counties_in_merged), 'State'] = pivoted_df['State'].str[:2]
print(pivoted_df[pivoted_df['State'].str.contains('\*')])
counties_without_plus = [county for county in counties_not_in_merged if '+' not in county]
print(counties_without_plus)
similar_counties = {}

# Finding similar counties with a lower cutoff
for county in counties_not_in_merged:
    matches = get_close_matches(county, merged_df['County'],n=2,cutoff=0.6)
    if matches:
        similar_counties[county] = matches[0]

for county in similar_counties.keys():
    print(county+": "+ similar_counties[county])

# Printing those that dont have plusses
for county in similar_counties.keys():
    if '+' not in county:
        print(county+": "+ similar_counties[county])

# Dropping Prince of Wales-Outer Ketchikan from pivoted_df
print(merged_df.loc[merged_df['County'].str.contains('Ketchikan'),'County'])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Ketchikan')])
pivoted_df = pivoted_df[pivoted_df['County'] != 'Prince of Wales-Outer Ketchikan']
edited_counties = edited_counties[edited_counties['County'] != 'Prince of Wales-Outer Ketchikan']
counties_not_in_merged = counties_not_in_merged[counties_not_in_merged != ('Prince of Wales-Outer Ketchikan')]
similar_counties = {county: similar_counties[county] for county in similar_counties if county != 'Prince of Wales-Outer Ketchikan'}

# Further cleaning
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Wrangell-Petersburg')]
print(merged_df.loc[merged_df['County'].str.contains('Chugach'),'County'])
print(merged_df.loc[merged_df['County'].str.contains('River'),['County','State']])
print(pivoted_df.loc[pivoted_df['County'].str.contains('River')])

# Dropping Chugach and Copper River from pivoted_df (They are listed under valdez-cordova)
#pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Chugach')]
#pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Copper river')]
#edited_counties = edited_counties[~edited_counties['County'].str.contains('Chugach')]
#edited_counties = edited_counties[~edited_counties['County'].str.contains('Copper river')]
#similar_counties = {county: similar_counties[county] for county in similar_counties if county != 'Chugach' and county != 'Copper river'}

# Changing wrangell to Wrangell City and Borough in pivoted df to match merged df.
print(merged_df[merged_df['County'].str.contains('Wrangell')])
print(pivoted_df[pivoted_df['County'].str.contains('Wrangell')])
pivoted_df.loc[pivoted_df['County'] == 'Wrangell', 'County'] = 'Wrangell City and Borough'
edited_counties.loc[edited_counties['County'] == 'Wrangell', 'County'] = 'Wrangell City and Borough'

####### Re-running prints of different combinations of counties with/without asterisks and plusses (After removing some counties)
counties_not_in_merged = []
edited_counties.head(len(edited_counties))

for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        counties_not_in_merged.append(county)
print(counties_not_in_merged)
counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(counties_in_merged)

pivoted_df.loc[pivoted_df['County'].isin(counties_in_merged), 'State'] = pivoted_df['State'].str[:2]
print(pivoted_df[pivoted_df['State'].str.contains('\*')])
counties_without_plus = [county for county in counties_not_in_merged if '+' not in county]
print(counties_without_plus)

similar_counties_new = {}
# Finding similar counties with a lower cutoff
for county in counties_not_in_merged:
    matches = get_close_matches(county, merged_df['County'],n=2,cutoff=0.6)
    if matches:
        similar_counties[county] = matches[0]

for county in similar_counties.keys():
    print(county+": "+ similar_counties[county])

# Printing those that dont have plusses
for county in similar_counties.keys():
    if '+' not in county:
        print(county+": "+ similar_counties[county])

# Fixing the Valdez-Cordova issue (Chugach and Copper River)
print(pivoted_df.loc[pivoted_df['County'].str.contains('Ketchikan'),'County'])
print(pivoted_df[pivoted_df['County'].isin(similar_counties.keys())])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Valdez')|pivoted_df['County'].str.contains('Copper')|pivoted_df['County'].str.contains('Chugach')])
print(merged_df.loc[merged_df['County'].str.contains('Valdez')|merged_df['County'].str.contains('Copper')|merged_df['County'].str.contains('Chugach')])
chugach_sum = pivoted_df.loc[pivoted_df['County'] == 'Chugach', 'Thousands of dollars'].sum()
copper_river_sum = pivoted_df.loc[pivoted_df['County'] == 'Copper River', 'Thousands of dollars'].sum()
pivoted_df.loc[pivoted_df['County'] == 'Valdez-Cordova', 'Thousands of dollars'] = int(chugach_sum) + int(copper_river_sum)

# Finding the gdp of Valdez-Cordova in chained 2012 dollars
# - Found by averaging the chained dollar index of other counties in Alaska from 2021 and dividing GDP by it.
pivoted_df.loc[pivoted_df['County'] == 'Valdez-Cordova', 'Thousands of chained 2012 dollars'] = 1783407

pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Chugach')]
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Copper River')]

print(pivoted_df.loc[pivoted_df['Quantity index'].str.contains('NA')])
print(pivoted_df.loc[pivoted_df['Quantity index'].isna()])

#### STILL NEED TO FIGURE OUT WHAT TO DO ABOUT VALDEZ-CORDOVA QUANTITY INDEX


#############################################################################################
#################### Re-running counties not in merged code #################################

counties_not_in_gdp = set(merged_df['County']) - set(pivoted_df['County'])
counties_not_in_gdp = sorted(counties_not_in_gdp)
print("Counties in merged_df but not in gdp_df:")
for county in counties_not_in_gdp:
    print(county)

counties_not_in_merged = set(pivoted_df['County']) - set(merged_df['County'])
counties_not_in_merged = sorted(counties_not_in_merged)
print("Counties in gdp but not in merged (sorted alphabetically):")
for county in counties_not_in_merged:
    print(county)


## Finding rows in pivoted_df with asterisks in the 'State' column
print(pivoted_df[pivoted_df['State'].str.contains('\*')])
edited_counties = pivoted_df[pivoted_df['State'].str.contains('\*')]
edited_counties.head(len(edited_counties))

# Using that list, finding counties that are in the merged_df/not in merged df
ast_counties_not_in_merged = []
for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        ast_counties_not_in_merged.append(county)
print(ast_counties_not_in_merged)

ast_counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(ast_counties_in_merged)

#############################################################################################

### Cleaning from top down of above results.

## Ketchikan Gateway
print(merged_df.loc[merged_df['County'].str.contains('Ketchikan'),'County'])
print(pivoted_df[pivoted_df['County'].str.contains('Ketchikan')])
ketch_similar_counties = get_close_matches('Ketchikan', merged_df['County'], n=5, cutoff=0.6)
print(ketch_similar_counties)
print(merged_df.loc[merged_df['State'].str.contains('AK'),'County'])
# - Ketchikan Gateway is included in Prince of Wales-Hyder Census Area in the merged_df

# Added Prince of Wales-Hyder GDP 
# -used it's chained 2012 dollars gdp and the price conversion factor caluculated earlier for Valdez-Cordova
pivoted_df.loc[pivoted_df['County'] == 'Prince of Wales-Hyder', 'Thousands of dollars'] = 298356
pivoted_df.loc[pivoted_df['County'] == 'Prince of Wales-Hyder', 'Thousands of dollars'] = pivoted_df.loc[pivoted_df['County']== 'Prince of Wales-Hyder', 'Thousands of dollars'].astype(int).sum() + pivoted_df.loc[pivoted_df['County'] == 'Ketchikan Gateway','Thousands of dollars'].astype(int).sum()
pivoted_df.loc[pivoted_df['County'] == 'Prince of Wales-Hyder', 'Thousands of chained 2012 dollars'] = pivoted_df.loc[pivoted_df['County'] == 'Prince of Wales-Hyder', 'Thousands of chained 2012 dollars'].astype(int).sum() + pivoted_df.loc[pivoted_df['County'] == 'Ketchikan Gateway', 'Thousands of chained 2012 dollars'].astype(int).sum()
pivoted_df.loc[pivoted_df['County'] == 'Prince of Wales-Hyder', 'Quantity index'] = pivoted_df.loc[pivoted_df['County'] == 'Prince of Wales-Hyder', 'Quantity index'].astype(float).sum() + pivoted_df.loc[pivoted_df['County'] == 'Ketchikan Gateway', 'Quantity index'].astype(float).sum()
print(pivoted_df.loc[pivoted_df['County'] == 'Prince of Wales-Hyder'])
# Dropped Ketchikan gateway from pivoted_df
pivoted_df = pivoted_df[pivoted_df['County'] != 'Ketchikan gateway']


## Kusilvak
print(merged_df.loc[merged_df['County'].str.contains('Kusilvak'),'County'])
print(pivoted_df[pivoted_df['County'].str.contains('Kusilvak')])
kus_similar_counties = get_close_matches('Kusilvak', merged_df['County'], n=5, cutoff=0.6)
print(kus_similar_counties)
print(merged_df.loc[merged_df['State'].str.contains('AK'),'County'])
print(merged_df.loc[merged_df['County'].str.contains('Hampton'),['County','State']])
# Cannot find a Kusilvak in the merged_df, so dropping it from pivoted_df
pivoted_df = pivoted_df[pivoted_df['County'] != 'Kusilvak']


## Yakutat
print(merged_df.loc[merged_df['County'].str.contains('Yakutat'),'County'])
print(pivoted_df[pivoted_df['County'].str.contains('Yakutat')])
yak_similar_counties = get_close_matches('Yakutat', merged_df['County'], n=5, cutoff=0.6)
print(yak_similar_counties)
print(merged_df.loc[merged_df['State'].str.contains('AK'),'County'])
print(merged_df.loc[merged_df['County'].str.contains('Hampton'),['County','State']])
# Adding Yakutat to Hoonah-Angoon because it is most likely represented by that in the merged_df
pivoted_df.loc[pivoted_df['County'] == 'Hoonah-Angoon', 'Thousands of dollars'] += pivoted_df.loc[pivoted_df['County'] == 'Yakutat', 'Thousands of dollars'].sum()
pivoted_df.loc[pivoted_df['County'] == 'Hoonah-Angoon', 'Thousands of chained 2012 dollars'] += pivoted_df.loc[pivoted_df['County'] == 'Yakutat', 'Thousands of chained 2012 dollars'].sum()
pivoted_df.loc[pivoted_df['County'] == 'Hoonah-Angoon', 'Quantity index'] += pivoted_df.loc[pivoted_df['County'] == 'Yakutat', 'Quantity index'].sum()
pivoted_df = pivoted_df[pivoted_df['County'] != 'Yakutat']

## Maui + Kalawao
county = 'Maui'
print(merged_df.loc[merged_df['County'].str.contains(county)])
print(merged_df.loc[merged_df['County'].str.contains('Kalawao')])
print(pivoted_df[pivoted_df['County'].str.contains(county)])
kus_similar_counties = get_close_matches(county, merged_df['County'], n=5, cutoff=0.6)
print(kus_similar_counties)

# Adding Maui and Kalawao values together in merged_df
maui_values = merged_df.loc[merged_df['County'].str.contains('Maui')]
kalawao_values = merged_df.loc[merged_df['County'].str.contains('Kalawao')]
colnames = merged_df.columns.tolist()
print(colnames)
print(merged_df[['FIPS','County','State']])
print(pivoted_df[['GeoFIPS','County','State']])
print(merged_df['FIPS'].isna().sum())

colnames = [col for col in colnames if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]
print(colnames)

for col in colnames:
    maui_values[col] += kalawao_values[col].sum()

print(maui_values[colnames])
print(maui_values)
print(merged_df.loc[merged_df['County'].str.contains('Maui')])
print(merged_df.loc[merged_df['County'].str.contains('Kalawao')])
merged_df.loc[merged_df['County'].str.contains('Maui')] = maui_values
merged_df = merged_df[~merged_df['County'].str.contains('Kalawao')]
merged_df.loc[merged_df['County'].str.contains('Maui'), 'County'] = 'Maui + Kalawao'

#Merging Maui and Kalawaoi in merged_df
print(merged_df.loc[merged_df['County'].str.contains('Maui')])

####
## Trying a more efficient approach to merging the counties in merged_df
combined_counties = [
    'Maui + Kalawao', 'Albemarle + Charlottesville', 'Alleghany + Covington',
    'Campbell + Lynchburg', 'Carroll + Galax', 'Frederick + Winchester',
    'Greensville + Emporia', 'Henry + Martinsville', 'James City + Williamsburg',
    'Montgomery + Radford', 'Pittsylvania + Danville', 'Prince George + Hopewell',
    'Roanoke + Salem', 'Rockingham + Harrisonburg', 'Southampton + Franklin',
    'Spotsylvania + Fredericksburg', 'Washington + Bristol', 'Wise + Norton', 
    'York + Poquoson'
]

# Split combined counties into two separate columns
split_counties = [item.split(' + ') for item in combined_counties]

# Create a DataFrame with the split counties
split_df = pd.DataFrame(split_counties, columns=['County1', 'County2'])
print(split_df)

# Filter the split_df to include only those rows where both counties are in merged_df
filtered_df = split_df[
    split_df['County1'].isin(merged_df['County']) &
    split_df['County2'].isin(merged_df['County'])
]

# Display the filtered combinations
print(filtered_df)

# Loop through each row in filtered_df to combine the counties in merged_df
for index, row in filtered_df.iterrows():
    county1 = row['County1']
    county2 = row['County2']
    
    print(f"Combining {county1} + {county2}")
    
    # Select the rows for each county
    county1_values = merged_df.loc[merged_df['County'].str.contains(county1)]
    county2_values = merged_df.loc[merged_df['County'].str.contains(county2)]
    
    # Ensure that the columns to be combined are numeric
    colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]
    
    # Combine the values from county2 into county1
    for col in colnames:
        if col in county1_values.columns and col in county2_values.columns:
            county1_values[col] += county2_values[col].sum()
    
    # Update the county1 row with the combined values
    merged_df.loc[merged_df['County'].str.contains(county1)] = county1_values
    
    # Remove the county2 row from merged_df
    merged_df = merged_df[~merged_df['County'].str.contains(county2)]
    
    # Rename the county1 row to reflect the combined counties
    merged_df.loc[merged_df['County'].str.contains(county1), 'County'] = f"{county1} + {county2}"

# Check the final merged_df
print(merged_df)

pd.set_option('display.max_rows', None)
print(merged_df[merged_df.isna()].sum())


#############################################################################################
#################### Re-running counties not in merged code #################################

counties_not_in_gdp = set(merged_df['County']) - set(pivoted_df['County'])
counties_not_in_gdp = sorted(counties_not_in_gdp)
print("Counties in merged_df but not in gdp_df:")
for county in counties_not_in_gdp:
    print(county)

counties_not_in_merged = set(pivoted_df['County']) - set(merged_df['County'])
counties_not_in_merged = sorted(counties_not_in_merged)
print("Counties in gdp but not in merged (sorted alphabetically):")
for county in counties_not_in_merged:
    print(county)


## Finding rows in pivoted_df with asterisks in the 'State' column
print(pivoted_df[pivoted_df['State'].str.contains('\*')])
edited_counties = pivoted_df[pivoted_df['State'].str.contains('\*')]
edited_counties.head(len(edited_counties))

# Using that list, finding counties that are in the merged_df/not in merged df
ast_counties_not_in_merged = []
for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        ast_counties_not_in_merged.append(county)
print(ast_counties_not_in_merged)

ast_counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(ast_counties_in_merged)

#############################################################################################

## Trying to figure out the problem with Ketchikan Gateway
print(pivoted_df.loc[pivoted_df['County'] == 'Ketchikan Gateway'])
print(merged_df.loc[merged_df['County'] == 'Ketchikan Gateway'])
merged_df = merged_df[~merged_df['County'].str.contains('Ketchikan Gateway')]
print(merged_df.loc[merged_df['County'].str.contains('Ketchikan Gateway')])
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Ketchikan Gateway')]
print(pivoted_df.loc[pivoted_df['County'].str.contains('Ketchikan Gateway')])

## Albermarle + Charlottseville
print(merged_df.loc[merged_df['County'].str.contains('Albemarle')])
print(merged_df.loc[merged_df['County'].str.contains('Charlottesville')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Albemarle')])
# Combine Albermarle and Charlottseville in merged_df
albermarle_values = merged_df.loc[merged_df['County'].str.contains('Albemarle')].copy()
charlottseville_values = merged_df.loc[merged_df['County'].str.contains('Charlottesville')].copy()

# Ensure that the columns to be combined are numeric
colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# Combine the values from Charlottseville into Albermarle
for col in colnames:
    albermarle_values[col] =albermarle_values[col].sum()+ charlottseville_values[col].sum()
    print(albermarle_values[col])

# Update the Albermarle row with the combined values
merged_df.loc[merged_df['County'].str.contains('Albemarle'), colnames] = albermarle_values[colnames]

# Remove the Charlottseville row from merged_df
merged_df = merged_df[~merged_df['County'].str.contains('Charlottesville')]

# Rename the Albermarle row to reflect the combined counties
merged_df.loc[merged_df['County'].str.contains('Albemarle'), 'County'] = 'Albemarle + Charlottesville'

# Check the final merged_df
print(merged_df.loc[merged_df['County'].str.contains('Albemarle')])
print(merged_df.loc[merged_df['County'].str.contains('Charlottesville')])   







######################

##### Trying to add/Fix GeoFIPS Column to be unique ID for the tables.

pivoted_df.head(20)
#Stopping point
pivoted_df.to_csv("PivotedData-Partial.csv", index=False)



geo_dict = pivoted_df.set_index('GeoFIPS')[['State', 'County']].to_dict('index')

# Convert the values from dictionaries to tuples
geo_dict = {k: (v['State'], v['County']) for k, v in geo_dict.items()}
print(geo_dict)

# Create a new column 'GeoTuple' in merged_df
merged_df['GeoFIPS'] = merged_df.apply(lambda x: geo_dict.get((x['County'], x['State']), ''), axis=1)
merged_df.head(20)
print(merged_df.isna().sum())





# Trying to fix GeoFIPS dict problems
#with open('/mnt/data/geo_data.csv', mode='w', newline='') as file:
    #writer = csv.writer(file)
    # Write the header row
    #writer.writerow(["Code", "State", "County"])
    
    # Write the data rows
    #for code, (state, county) in geo_dict.items():
    #    writer.writerow([code.strip(), state, county.strip()])