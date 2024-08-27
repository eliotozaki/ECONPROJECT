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


#############################################################################################
## Trying a more efficient approach to merging the counties in merged_df
combined_counties = pivoted_df.loc[pivoted_df['State'].str.endswith('*'), ['County', 'State']]
print(combined_counties)

############
## Doing Maui + Kalawao first

print(merged_df.loc[merged_df['County'].str.contains('Maui')])
print(merged_df.loc[merged_df['County'].str.contains('Kalawao')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Maui')])


maui_values = merged_df.loc[merged_df['County'].str.contains('Maui')].copy()
kalawao_values = merged_df.loc[merged_df['County'].str.contains('Kalawao')].copy()

# Ensure that the columns to be combined are numeric
colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# Combine the values from Kalawao into Maui
for col in colnames:
    maui_values[col] = maui_values[col].sum() + kalawao_values[col].sum()

# Update the Maui row with the combined values
merged_df.loc[merged_df['County'].str.contains('Maui'), colnames] = maui_values[colnames]

# Remove the Kalawao row from merged_df
merged_df = merged_df[~merged_df['County'].str.contains('Kalawao')]

# Rename the Maui row to reflect the combined counties
merged_df.loc[merged_df['County'].str.contains('Maui'), 'County'] = 'Maui + Kalawao'

# Check the final merged_dfunty2 = row['County2']
print(merged_df.loc[merged_df['County'].str.contains('Maui')])
print(merged_df.loc[merged_df['County'].str.contains('Kalawao')])



######## Now doing the rest, which are all in VA
combined_counties = combined_counties[2:]
print(combined_counties['County'])


for county in combined_counties['County']:
    county1, county2 = county.split(' + ')
    print(f"Combining {county1} + {county2}")
    
    county1_values = merged_df.loc[merged_df['County'].str.contains(county1)& merged_df['State'].str.contains('VA')].copy()
    county2_values = merged_df.loc[merged_df['County'].str.contains(county2) & merged_df['State'].str.contains('VA')].copy()
    
    for col in colnames:
        if col in county1_values.columns and col in county2_values.columns:
            county1_values[col] += county2_values[col].sum()
    
    merged_df.loc[merged_df['County'].str.contains(county1)& merged_df['State'].str.contains('VA') ] = county1_values
    merged_df = merged_df[~(merged_df['County'].str.contains(county2)& merged_df['State'].str.contains('VA'))]
    merged_df.loc[merged_df['County'].str.contains(county1)& merged_df['State'].str.contains('VA'), 'County'] = f"{county1} + {county2}"
    print(merged_df.loc[merged_df['County'].str.contains(county2)])

## Trying to figure out the problem with Ketchikan Gateway
print(pivoted_df.loc[pivoted_df['County'] == 'Ketchikan Gateway'])
print(merged_df.loc[merged_df['County'] == 'Ketchikan Gateway'])
merged_df = merged_df[~merged_df['County'].str.contains('Ketchikan Gateway')]
print(merged_df.loc[merged_df['County'].str.contains('Ketchikan Gateway')])
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Ketchikan Gateway')]
print(pivoted_df.loc[pivoted_df['County'].str.contains('Ketchikan Gateway')])

# #############################################################################################
# ################################## UNUSED (OUTDATED) ########################################
# combined_counties = [
#     'Maui + Kalawao', 'Albemarle + Charlottesville', 'Alleghany + Covington',
#     'Campbell + Lynchburg', 'Carroll + Galax', 'Frederick + Winchester',
#     'Greensville + Emporia', 'Henry + Martinsville', 'James City + Williamsburg',
#     'Montgomery + Radford', 'Pittsylvania + Danville', 'Prince George + Hopewell',
#     'Roanoke + Salem', 'Rockingham + Harrisonburg', 'Southampton + Franklin',
#     'Spotsylvania + Fredericksburg', 'Wise + Norton', 
#     'York + Poquoson'
# ]

# # Using that list, finding counties that are in the merged_df/not in merged df
# combined_counties_states = []
# for county in combined_counties:
#     combined_counties_states.append(pivoted_df.loc[pivoted_df['County'].str.contains(county), 'State'])

# # Split combined counties into two separate columns
# split_counties = [item.split(' + ') for item in combined_counties]

# # Create a DataFrame with the split counties
# split_df = pd.DataFrame(split_counties, columns=['County1', 'County2'])
# print(split_df)

# # Filter the split_df to include only those rows where both counties are in merged_df
# filtered_df = split_df[
#     split_df['County1'].isin(merged_df['County']) &
#     split_df['County2'].isin(merged_df['County'])
# ]

# # Display the filtered combinations
# print(filtered_df)

# # Loop through each row in filtered_df to combine the counties in merged_df
# for index, row in filtered_df.iterrows():
#     county1 = row['County1']
#     county2 = row['County2']
    
#     print(f"Combining {county1} + {county2}")
    
#     # Select the rows for each county
#     county1_values = merged_df.loc[merged_df['County'].str.contains(county1)]
#     county2_values = merged_df.loc[merged_df['County'].str.contains(county2)]
    
#     # Ensure that the columns to be combined are numeric
#     colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]
    
#     # Combine the values from county2 into county1
#     for col in colnames:
#         if col in county1_values.columns and col in county2_values.columns:
#             county1_values[col] += county2_values[col].sum()
    
#     # Update the county1 row with the combined values
#     merged_df.loc[merged_df['County'].str.contains(county1)] = county1_values
    
#     # Remove the county2 row from merged_df
#     merged_df = merged_df[~merged_df['County'].str.contains(county2)]
    
#     # Rename the county1 row to reflect the combined counties
#     merged_df.loc[merged_df['County'].str.contains(county1), 'County'] = f"{county1} + {county2}"

# # Check the final merged_df
# print(merged_df)

# pd.set_option('display.max_rows', None)
# print(merged_df[merged_df.isna()].sum())


# #############################################################################################
# #################### Re-running counties not in merged code #################################

# counties_not_in_gdp = set(merged_df['County']) - set(pivoted_df['County'])
# counties_not_in_gdp = sorted(counties_not_in_gdp)
# print("Counties in merged_df but not in gdp_df:")
# for county in counties_not_in_gdp:
#     print(county)

# counties_not_in_merged = set(pivoted_df['County']) - set(merged_df['County'])
# counties_not_in_merged = sorted(counties_not_in_merged)
# print("Counties in gdp but not in merged (sorted alphabetically):")
# for county in counties_not_in_merged:
#     print(county)


# ## Finding rows in pivoted_df with asterisks in the 'State' column
# print(pivoted_df[pivoted_df['State'].str.contains('\*')])
# edited_counties = pivoted_df[pivoted_df['State'].str.contains('\*')]
# edited_counties.head(len(edited_counties))

# # Using that list, finding counties that are in the merged_df/not in merged df
# ast_counties_not_in_merged = []
# for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
#     if (county, state) not in zip(merged_df['County'], merged_df['State']):
#         ast_counties_not_in_merged.append(county)
# print(ast_counties_not_in_merged)

# ast_counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
# print(ast_counties_in_merged)

# #############################################################################################
# ###############
# ## Albermarle + Charlottseville
# print(merged_df.loc[merged_df['County'].str.contains('Albemarle')])
# print(merged_df.loc[merged_df['County'].str.contains('Charlottesville')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Albemarle')])
# # Combine Albermarle and Charlottseville in merged_df
# albermarle_values = merged_df.loc[merged_df['County'].str.contains('Albemarle')].copy()
# charlottseville_values = merged_df.loc[merged_df['County'].str.contains('Charlottesville')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Charlottseville into Albermarle
# for col in colnames:
#     albermarle_values[col] =albermarle_values[col].sum()+ charlottseville_values[col].sum()
#     print(albermarle_values[col])

# # Update the Albermarle row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Albemarle'), colnames] = albermarle_values[colnames]

# # Remove the Charlottseville row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Charlottesville')]

# # Rename the Albermarle row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Albemarle'), 'County'] = 'Albemarle + Charlottesville'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Albemarle')])
# print(merged_df.loc[merged_df['County'].str.contains('Charlottesville')])   

# ###############
# ## Campbell + Lynchburg
# print(merged_df.loc[merged_df['County'].str.contains('Campbell')])
# print(merged_df.loc[merged_df['County'].str.contains('Lynchburg')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Campbell')])
# # Combine Campbell and Lynchburg in merged_df
# campbell_values = merged_df.loc[merged_df['County'].str.contains('Campbell') & merged_df['State'].str.contains('VA')].copy()
# lynchburg_values = merged_df.loc[merged_df['County'].str.contains('Lynchburg')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Lynchburg into Campbell
# for col in colnames:
#     campbell_values[col] = campbell_values[col].sum() + lynchburg_values[col].sum()
#     print(campbell_values[col])

# # Update the Campbell row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Campbell') & merged_df['State'].str.contains('VA'), colnames] = campbell_values[colnames]

# # Remove the Lynchburg row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Lynchburg')]

# # Rename the Campbell row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Campbell') & merged_df['State'].str.contains('VA'), 'County'] = 'Campbell + Lynchburg'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Campbell')])
# print(merged_df.loc[merged_df['County'].str.contains('Lynchburg')])   

# ########## 
# ## Carroll + Galax
# print(merged_df.loc[merged_df['County'].str.contains('Carroll')])
# print(merged_df.loc[merged_df['County'].str.contains('Galax')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Carroll')])
# # Combine Carroll and Galax in merged_df
# carroll_values = merged_df.loc[merged_df['County'].str.contains('Carroll')& merged_df['State'].str.contains('VA')].copy()
# galax_values = merged_df.loc[merged_df['County'].str.contains('Galax')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Galax into Carroll
# for col in colnames:
#     carroll_values[col] = carroll_values[col].sum() + galax_values[col].sum()
#     print(carroll_values[col])

# # Update the Carroll row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Carroll')& merged_df['State'].str.contains('VA'), colnames] = carroll_values[colnames]

# # Remove the Galax row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Galax')]

# # Rename the Carroll row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Carroll')& merged_df['State'].str.contains('VA'), 'County'] = 'Carroll + Galax'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Carroll')])
# print(merged_df.loc[merged_df['County'].str.contains('Galax')])   


# ######################
# ## Frederick + Winchester
# print(merged_df.loc[merged_df['County'].str.contains('Frederick')])
# print(merged_df.loc[merged_df['County'].str.contains('Winchester')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Frederick')])
# # Combine Frederick and Winchester in merged_df
# # Frederick county, VA
# frederick_values = merged_df.loc[merged_df['County'] == ('Frederick') & merged_df['State'].str.contains('VA')].copy()
# winchester_values = merged_df.loc[merged_df['County'].str.contains('Winchester')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Winchester into Frederick
# for col in colnames:
#     frederick_values[col] = frederick_values[col].sum() + winchester_values[col].sum()
#     print(frederick_values[col])

# # Update the Frederick row with the combined values
# merged_df.loc[merged_df['County']==('Frederick')& merged_df['State'].str.contains('VA'), colnames] = frederick_values[colnames]

# # Remove the Winchester row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Winchester')]

# # Rename the Frederick row to reflect the combined counties
# merged_df.loc[(merged_df['County']=='Frederick') & merged_df['State'].str.contains('VA'), 'County'] = 'Frederick + Winchester'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Frederick')])
# print(merged_df.loc[merged_df['County'].str.contains('Winchester')])  



# ######################
# ## Greensville + Emporia
# print(merged_df.loc[merged_df['County'].str.contains('Greensville')])
# print(merged_df.loc[merged_df['County'].str.contains('Emporia')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Greensville')])
# # Combine Greensville and Emporia in merged_df
# greensville_values = merged_df.loc[merged_df['County'].str.contains('Greensville')].copy()
# emporia_values = merged_df.loc[merged_df['County'].str.contains('Emporia')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Emporia into Greensville
# for col in colnames:
#     greensville_values[col] = greensville_values[col].sum() + emporia_values[col].sum()
#     print(greensville_values[col])

# # Update the Greensville row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Greensville'), colnames] = greensville_values[colnames]

# # Remove the Emporia row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Emporia')]

# # Rename the Greensville row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Greensville'), 'County'] = 'Greensville + Emporia'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Greensville')])
# print(merged_df.loc[merged_df['County'].str.contains('Emporia')])  

# ######################
# ## Henry + Martinsville
# print(merged_df.loc[merged_df['County'].str.contains('Henry')])
# print(merged_df.loc[merged_df['County'].str.contains('Martinsville')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Henry')])
# # Combine Henry and Martinsville in merged_df
# henry_values = merged_df.loc[merged_df['County'].str.contains('Henry') & merged_df['State'].str.contains('VA')].copy()
# martinsville_values = merged_df.loc[merged_df['County'].str.contains('Martinsville')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Martinsville into Henry
# for col in colnames:
#     henry_values[col] = henry_values[col].sum() + martinsville_values[col].sum()
#     print(henry_values[col])

# # Update the Henry row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Henry') & merged_df['State'].str.contains('VA'), colnames] = henry_values[colnames]

# # Remove the Martinsville row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Martinsville')]

# # Rename the Henry row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Henry') & merged_df['State'].str.contains('VA'), 'County'] = 'Henry + Martinsville'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Henry')])
# print(merged_df.loc[merged_df['County'].str.contains('Martinsville')])  


# ######################
# ## Montgomery + Radford

# print(merged_df.loc[merged_df['County'].str.contains('Montgomery')])
# print(merged_df.loc[merged_df['County'].str.contains('Radford')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Montgomery')])

# # Combine Montgomery and Radford in merged_df
# montgomery_values = merged_df.loc[merged_df['County'].str.contains('Montgomery') & merged_df['State'].str.contains('VA')].copy()
# radford_values = merged_df.loc[merged_df['County'].str.contains('Radford')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Radford into Montgomery
# for col in colnames:
#     montgomery_values[col] = montgomery_values[col].sum() + radford_values[col].sum()
#     print(montgomery_values[col])

# # Update the Montgomery row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Montgomery') & merged_df['State'].str.contains('VA'), colnames] = montgomery_values[colnames]

# # Remove the Radford row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Radford')]

# # Rename the Montgomery row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Montgomery') & merged_df['State'].str.contains('VA'), 'County'] = 'Montgomery + Radford'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Montgomery')])
# print(merged_df.loc[merged_df['County'].str.contains('Radford')])



# ######################
# ## Pittsylvania + Danville
# print(merged_df.loc[merged_df['County'].str.contains('Pittsylvania')])
# print(merged_df.loc[merged_df['County'].str.contains('Danville')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Pittsylvania')])

# # Combine Pittsylvania and Danville in merged_df
# pittsylvania_values = merged_df.loc[merged_df['County'].str.contains('Pittsylvania') & merged_df['State'].str.contains('VA')].copy()
# danville_values = merged_df.loc[merged_df['County'].str.contains('Danville')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Danville into Pittsylvania
# for col in colnames:
#     pittsylvania_values[col] = pittsylvania_values[col].sum() + danville_values[col].sum()
#     print(pittsylvania_values[col])

# # Update the Pittsylvania row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Pittsylvania') & merged_df['State'].str.contains('VA'), colnames] = pittsylvania_values[colnames]

# # Remove the Danville row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Danville')]

# # Rename the Pittsylvania row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Pittsylvania') & merged_df['State'].str.contains('VA'), 'County'] = 'Pittsylvania + Danville'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Pittsylvania')])
# print(merged_df.loc[merged_df['County'].str.contains('Danville')])

# ####################################
# ## Prince George + Hopewell

# print(merged_df.loc[merged_df['County'].str.contains('Prince George')])
# print(merged_df.loc[merged_df['County'].str.contains('Hopewell')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Prince George')])

# # Combine Prince George and Hopewell in merged_df
# prince_george_values = merged_df.loc[merged_df['County'].str.contains('Prince George') & merged_df['State'].str.contains('VA')].copy()
# hopewell_values = merged_df.loc[merged_df['County'].str.contains('Hopewell')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Hopewell into Prince George
# for col in colnames:
#     prince_george_values[col] = prince_george_values[col].sum() + hopewell_values[col].sum()
#     print(prince_george_values[col])

# # Update the Prince George row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Prince George') & merged_df['State'].str.contains('VA'), colnames] = prince_george_values[colnames]

# # Remove the Hopewell row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Hopewell')]

# # Rename the Prince George row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Prince George') & merged_df['State'].str.contains('VA'), 'County'] = 'Prince George + Hopewell'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Prince George')])
# print(merged_df.loc[merged_df['County'].str.contains('Hopewell')])


# ####################################
# ## Rockingham + Harrisonburg

# print(merged_df.loc[merged_df['County'].str.contains('Rockingham')])
# print(merged_df.loc[merged_df['County'].str.contains('Harrisonburg')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Rockingham')])

# # Combine Rockingham and Harrisonburg in merged_df
# rockingham_values = merged_df.loc[merged_df['County'].str.contains('Rockingham') & merged_df['State'].str.contains('VA')].copy()
# harrisonburg_values = merged_df.loc[merged_df['County'].str.contains('Harrisonburg')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Harrisonburg into Rockingham
# for col in colnames:
#     rockingham_values[col] = rockingham_values[col].sum() + harrisonburg_values[col].sum()
#     print(rockingham_values[col])

# # Update the Rockingham row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Rockingham') & merged_df['State'].str.contains('VA'), colnames] = rockingham_values[colnames]

# # Remove the Harrisonburg row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Harrisonburg')]

# # Rename the Rockingham row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Rockingham') & merged_df['State'].str.contains('VA'), 'County'] = 'Rockingham + Harrisonburg'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Rockingham')])
# print(merged_df.loc[merged_df['County'].str.contains('Harrisonburg')])


# ####################################
# ## York + Poquoson

# print(merged_df.loc[merged_df['County'].str.contains('York')])
# print(merged_df.loc[merged_df['County'].str.contains('Poquoson')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('York')])

# # Combine York and Poquoson in merged_df
# york_values = merged_df.loc[merged_df['County'].str.contains('York') & merged_df['State'].str.contains('VA')].copy()
# poquoson_values = merged_df.loc[merged_df['County'].str.contains('Poquoson')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Poquoson into York
# for col in colnames:
#     york_values[col] = york_values[col].sum() + poquoson_values[col].sum()
#     print(york_values[col])

# # Update the York row with the combined values
# merged_df.loc[merged_df['County'].str.contains('York') & merged_df['State'].str.contains('VA'), colnames] = york_values[colnames]

# # Remove the Poquoson row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Poquoson')]

# # Rename the York row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('York') & merged_df['State'].str.contains('VA'), 'County'] = 'York + Poquoson'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('York')])
# print(merged_df.loc[merged_df['County'].str.contains('Poquoson')])

# ###########################################
# ## Spotsylvania + Fredericksburg

# print(merged_df.loc[merged_df['County'].str.contains('Spotsylvania')])
# print(merged_df.loc[merged_df['County'].str.contains('Fredericksburg')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Spotsylvania')])

# # Combine Spotsylvania and Fredericksburg in merged_df
# spotsylvania_values = merged_df.loc[merged_df['County'].str.contains('Spotsylvania') & merged_df['State'].str.contains('VA')].copy()
# fredericksburg_values = merged_df.loc[merged_df['County'].str.contains('Fredericksburg')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# # Combine the values from Fredericksburg into Spotsylvania
# for col in colnames:
#     spotsylvania_values[col] = spotsylvania_values[col].sum() + fredericksburg_values[col].sum()
#     print(spotsylvania_values[col])

# # Update the Spotsylvania row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Spotsylvania') & merged_df['State'].str.contains('VA'), colnames] = spotsylvania_values[colnames]

# # Remove the Fredericksburg row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Fredericksburg')]

# # Rename the Spotsylvania row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Spotsylvania') & merged_df['State'].str.contains('VA'), 'County'] = 'Spotsylvania + Fredericksburg'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Spotsylvania')])
# print(merged_df.loc[merged_df['County'].str.contains('Fredericksburg')])



# ###########################################
# ## Washington + Bristol

# print(merged_df.loc[merged_df['County'].str.contains('Washington')])
# print(merged_df.loc[merged_df['County'].str.contains('Bristol')])
# print(pivoted_df.loc[pivoted_df['County'].str.contains('Washington')])
# # Combine Washington and Bristol in merged_df
# washington_values = merged_df.loc[merged_df['County'].str.contains('Washington') & merged_df['State'].str.contains('VA')].copy()
# bristol_values = merged_df.loc[merged_df['County'].str.contains('Bristol')].copy()

# # Ensure that the columns to be combined are numeric
# colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]
# print(colnames)

# # Combine the values from Bristol into Washington
# for col in colnames:
#     washington_values[col] = washington_values[col].sum() + bristol_values[col].sum()
#     print(washington_values[col])

# # Update the Washington row with the combined values
# merged_df.loc[merged_df['County'].str.contains('Washington') & merged_df['State'].str.contains('VA'), colnames] = washington_values[colnames]

# # Remove the Bristol row from merged_df
# merged_df = merged_df[~merged_df['County'].str.contains('Bristol')]
# print(merged_df.loc[merged_df['County'].str.contains('Bristol')])
# # Rename the Washington row to reflect the combined counties
# merged_df.loc[merged_df['County'].str.contains('Washington') & merged_df['State'].str.contains('VA'), 'County'] = 'Washington + Bristol'

# # Check the final merged_df
# print(merged_df.loc[merged_df['County'].str.contains('Washington')])
# print(merged_df.loc[merged_df['County'].str.contains('Bristol')])

# #############################################################################################
# #############################################################################################
# #############################################################################################
# #############################################################################################


###########################################################################################
## Final checks for combined counties in pivoted_df that arent in merged_df

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
for county in ast_counties_in_merged:
    merged_df.loc[merged_df['County'] == county, 'State'] = merged_df.loc[merged_df['County'] == county, 'State'].str.rstrip('*')


for county in ast_counties_in_merged:
    pivoted_df.loc[pivoted_df['County'] == county, 'State'] = pivoted_df.loc[pivoted_df['County'] == county, 'State'].str.rstrip('*')
print(pivoted_df.loc[pivoted_df['State'].str.contains('\*')])

# Editing Alutians West
pivoted_df.loc[pivoted_df['County'].str.contains('Aleutians E'), 'County'] = 'Aleutians East'

###########################################################################################

## Making unique identifier columns: GeoFIPS and making a useable "State and County" column 
print(merged_df.columns)
print(pivoted_df['State'].unique())


# Find counties and states from pivoted_df where the state value is not in the state values of merged_df
unique_states = pivoted_df.loc[~pivoted_df['State'].isin(merged_df['State']), ['County', 'State']]
print(unique_states)
print(pivoted_df.loc[pivoted_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['County']), ['County','State']])

pivoted_df.loc[pivoted_df['State'].isin(unique_states['State']), 'State'] = 'VA'

counties_with_trailing_space = merged_df.loc[merged_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])
counties_with_trailing_space = pivoted_df.loc[pivoted_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])

# Making the State and County Columns for both dataframes
merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']

merged_df[['State and County', 'State', 'County']].head(20)
pivoted_df[['State and County', 'State', 'County']].head(25)




### Creating a new stopping place
# Exporting the cleaned datasets
merged_df.to_csv("AllData-MergedDS.csv")
pivoted_df.to_csv("PivotedData-Cleaned.csv")

######################################################################################

### New Stopping place/Starting place
#Cleaning More Datasets
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap
from difflib import get_close_matches

#from pandas_profiling import ProfileReport

## Cleaning
# Load the datasets
population_df = pd.read_csv("Cleaned-Datasets\POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("Cleaned-Datasets/EMISSIONSDATA-Cleaned.csv")
merged_df = pd.read_csv("AllData-MergedDS.csv")
pivoted_df = pd.read_csv("PivotedData-Cleaned.csv")



## Checking which State and County values are in respective dfs and not in other.
counties_not_in_gdp = set(merged_df['State and County']) - set(pivoted_df['State and County'])
counties_not_in_gdp = sorted(counties_not_in_gdp)
print("Counties in merged_df but not in gdp_df:")
for county in counties_not_in_gdp:
    print(county)

counties_not_in_merged = set(pivoted_df['State and County']) - set(merged_df['State and County'])
counties_not_in_merged = sorted(counties_not_in_merged)
print("Counties in gdp but not in merged (sorted alphabetically):")
for county in counties_not_in_merged:
    print(county)

#print(merged_df.loc[merged_df['State and County'].str.contains('Franklin')])
#print(pivoted_df.loc[pivoted_df['State and County'].str.contains('Franklin')])

print(merged_df.loc[merged_df['County'].str.contains('Anchorage')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage')])
pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage'), 'County'] = 'Anchorage'
pivoted_df.loc[pivoted_df['State and County'].str.contains('Anchorage'), 'State and County'] = 'AK Anchorage'

similar_counties_new = {}
# Finding similar counties with a lower cutoff
for county in counties_not_in_merged:
    matches = get_close_matches(county, merged_df['State and County'], n=2, cutoff=0.8)
    if matches:
        similar_counties_new[county] = matches[0]
print(similar_counties_new)

### Counties to edit: IN Lagrange, MO Ste Genevieve, NM Dona Ana, TX Eastland
print(merged_df.loc[merged_df['County'].str.contains('LaGrange')])
pivoted_df.loc[pivoted_df['County'].str.contains('Lagrange'), 'County'] = 'LaGrange'
print(pivoted_df.loc[pivoted_df['County'].str.contains('LaGrange')])

print(merged_df.loc[merged_df['County'].str.contains('Ste Genevieve')])
merged_df.loc[merged_df['County'].str.contains('Ste Genevieve'), 'County'] = 'Ste. Genevieve'

print(pivoted_df.loc[pivoted_df['County'].str.contains('a Ana')])
pivoted_df.loc[pivoted_df['County'].str.contains('a Ana'), 'County'] = 'Dona Ana'

print(merged_df.loc[merged_df['County'].str.contains('Eastland')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Eland')])
pivoted_df.loc[pivoted_df['County'].str.contains('Eland'), 'County'] = 'Eastland'

# Making the State and County Columns for both dataframes
merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']

print(merged_df[['State and County', 'State', 'County']].head(20))
print(pivoted_df[['State and County', 'State', 'County']].head(25))

## Finding counties that are similar between counties not in merged and counties not in gdp
similar_counties_new = {}
# Finding similar counties with a lower cutoff
for county in counties_not_in_merged:
    matches = get_close_matches(county, counties_not_in_gdp, n=2, cutoff=0.5)
    if matches:
        similar_counties_new[county] = matches[0]
print(similar_counties_new)

## Carson City
print(merged_df.loc[merged_df['County'].str.contains('Carson City')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Carson City')])
pivoted_df.loc[pivoted_df['County'].str.contains('Carson City'), 'County'] = 'Carson City'

## Fremont
print(merged_df.loc[merged_df['County'].str.contains('Fremont')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Fremont')])
pivoted_df.loc[pivoted_df['County'].str.contains('Fremont')& pivoted_df['State'].str.contains('ID'), 'County'] = 'Fremont'

# Making the State and County Columns for both dataframes
merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


##############################################################################
## Re-running code to check for similar counties
## Checking which State and County values are in respective dfs and not in other.
counties_not_in_gdp = set(merged_df['State and County']) - set(pivoted_df['State and County'])
counties_not_in_gdp = sorted(counties_not_in_gdp)
print("Counties in merged_df but not in gdp_df:")
for county in counties_not_in_gdp:
    print(county)

counties_not_in_merged = set(pivoted_df['State and County']) - set(merged_df['State and County'])
counties_not_in_merged = sorted(counties_not_in_merged)
print("Counties in gdp but not in merged (sorted alphabetically):")
for county in counties_not_in_merged:
    print(county)

pivoted_VA = pivoted_df.loc[pivoted_df['State'].str.contains('VA')]

similar_counties_new = {}
# Finding similar counties with a lower cutoff
for county in counties_not_in_gdp:
    matches = get_close_matches(county, pivoted_VA['State and County'], n=2, cutoff=0.9)
    if matches:
        similar_counties_new[county] = matches[0]
print(similar_counties_new)

print(len(pivoted_VA))
print(len(merged_df[merged_df['State'] == 'VA']))



### Trying to finish counties_not_in_gdp manually

# After looking online, have the following info:
# Buena Vista is in Rockbridge Coutny
# Colonial Heights is is Dinwiddie County
# Fairfax City is in Fairfax County
# Falls Church is in Arlington County
# Lexington is in Rockbridge County
# Manassas/Manassas Park are in Prince William County
# City of Petersburg is in Dinwiddie County
# Staunton is in Augusta County
# Waynesboro is in Augusta County

# Falls Church might be in Arlington or Fairfax

print(merged_df.loc[merged_df['County'].str.contains('Rockbridge')])
print(merged_df.loc[merged_df['County'].str.contains('Buena Vista')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Buena Vista')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Rockbridge')])

original_counties = ['City of Buena Vista', 'City of Colonial Heights', 'City of Fairfax', 'City of Falls Church', 'City of Lexington', 'City of Manassas Park','City of Manassas', 'City of Petersburg', 'City of Staunton', 'City of Waynesboro']
equivalent_counties = ['Rockbridge', 'Dinwiddie', 'Fairfax', 'Arlington', 'Rockbridge', 'Prince William','Prince William', 'Dinwiddie', 'Augusta', 'Augusta']

print(merged_df.loc[merged_df['County'].str.contains('Arlington')])

for index in range(len(original_counties)):
    og = original_counties[index]
    eq = equivalent_counties[index]
    state = 'VA'
    og_val = merged_df.loc[merged_df['County'].str.contains(og) & merged_df['State'].str.contains(state)].copy()
    eq_val = merged_df.loc[merged_df['County'].str.contains(eq) & merged_df['State'].str.contains(state)].copy() 
     # Ensure that the columns to be combined are numeric
    colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]
        
    # Combine the values from eq_val into og_val
    for col in colnames:
        eq_val[col] = og_val[col].sum() + eq_val[col].sum()
        
        # Update the og_val row with the combined values
    merged_df.loc[merged_df['County'].str.contains(eq) & merged_df['State'].str.contains(state), colnames] = eq_val[colnames]
        
        # Remove the eq_val row from merged_df
    merged_df = merged_df[~(merged_df['County'].str.contains(og) & merged_df['State'].str.contains(state))]

print(merged_df.loc[merged_df['County'].str.contains('Rockbridge')])


##############################################################################
## Re-running code to check for similar counties
## Checking which State and County values are in respective dfs and not in other.
counties_not_in_gdp = set(merged_df['State and County']) - set(pivoted_df['State and County'])
counties_not_in_gdp = sorted(counties_not_in_gdp)
print("Counties in merged_df but not in gdp_df:")
for county in counties_not_in_gdp:
    print(county)

counties_not_in_merged = set(pivoted_df['State and County']) - set(merged_df['State and County'])
counties_not_in_merged = sorted(counties_not_in_merged)
print("Counties in gdp but not in merged (sorted alphabetically):")
for county in counties_not_in_merged:
    print(county)

merged_AK = merged_df.loc[merged_df['State'].str.contains('AK')]
pivoted_AK = pivoted_df.loc[pivoted_df['State'].str.contains('AK')]
print(len(merged_AK))
print(len(pivoted_AK))

for county in pivoted_AK['State and County']:
    if county in merged_AK['State and County']:
        print(county)
    else:
        print("NOT IN MERGED: ", county)


for county in merged_AK['State and County']:
    print(county)

for county in pivoted_AK:
    print(county)

pivoted_AK.head(len(pivoted_AK))
merged_AK.head(len(merged_AK))

## There is missing data from Carbon emissions in the merged_df's Alaska data. This should be mitigated in the ds by some merging of measurements from those unnacounted counties into the accounted counties.
## Solution would then be to delete the rows in Alaska for unnacounted counties in merged_df, 

pivoted_df = pivoted_df[~(pivoted_df['State and County'].str.contains('AK') & ~pivoted_df['County'].isin(merged_AK['County']))]

## Dealing with the rest of the counties not in merged

print(merged_df.loc[merged_df['County'].str.contains('Hartford')])



##########################################################################################################
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