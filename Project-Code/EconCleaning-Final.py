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
population_df = pd.read_csv("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Cleaned-Datasets/POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Cleaned-Datasets/EMISSIONSDATA-Cleaned.csv")

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
emissions_df = pd.read_csv("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Cleaned-Datasets/EMISSIONSDATA-Cleaned.csv")

population_df.head(100)
emissions_df.head(100)
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

emissions_df.columns
population_df.columns
population_df.head(10)
population_df.drop(['State and County'], axis=1, inplace=True)
population_df.drop(['Unnamed: 0'], axis=1, inplace=True)
emissions_df.drop(['Unnamed: 0'], axis=1, inplace=True)



# Making a unique identifier column for both datasets (State and County)
population_df['County'] = population_df['County'].str.rstrip()
emissions_df['County'] = emissions_df['County'].str.rstrip()
population_df['State'] = population_df['State'].str.rstrip()
emissions_df['State'] = emissions_df['State'].str.rstrip()

###
population_df['State and County'] = population_df['State'] + ' ' + population_df['County']
emissions_df['State and County'] = emissions_df['State'] + ' ' + emissions_df['County']

population_df.head(20)
emissions_df.head(20)

# Setting index to 'County' for both datasets
emissions_df.set_index("State and County")
population_df.set_index("State and County")

print(emissions_df.loc[emissions_df['County'].str.contains('Vald')])
print(population_df.loc[population_df['County'].str.contains('Vald')])


counties_not_in_merged = set(population_df['State and County']) - set(emissions_df['State and County'])
print("Counties in population_df but not in emissions:")
for county in counties_not_in_merged:
    print(county)

print(population_df.loc[population_df['State and County'].str.contains('Salle')])
print(merged_df.loc[merged_df['State and County'].str.contains('Salle')])
print(emissions_df.loc[emissions_df['State and County'].str.contains('Salle')])

emissions_df.loc[emissions_df['State and County'].str.contains('IL La Salle'), 'County'] = 'LaSalle'
population_df['State and County'] = population_df['State'] + ' ' + population_df['County']
emissions_df['State and County'] = emissions_df['State'] + ' ' + emissions_df['County']

print(emissions_df.loc[emissions_df['County'].str.contains('Claiborne')])
print(population_df.loc[population_df['County'].str.contains('Claiborne')])

print(len(population_df[population_df['State'] == 'LA']))
print(len(emissions_df[population_df['State']=='LA'] ))
for county in emissions_df.loc[emissions_df['State'].str.contains('LA'),'State and County']:
    print(county)

counties_not_in_emissions = set(emissions_df['State and County']) - set(population_df['State and County'])
print("Counties in emissions but not in population:")
for county in counties_not_in_emissions:
    print(county)

print(emissions_df.loc[emissions_df['State and County'].str.contains('LaSalle')])
print(population_df.loc[population_df['State and County'].str.contains('LaSalle')])

emissions_df.loc[emissions_df['State and County'].str.contains('LA LaSalle'), 'County'] = 'Claiborne'

population_df['State and County'] = population_df['State'] + ' ' + population_df['County']
emissions_df['State and County'] = emissions_df['State'] + ' ' + emissions_df['County']

counties_not_in_merged = set(population_df['State and County']) - set(emissions_df['State and County'])
print("Counties in population_df but not in emissions:")
for county in counties_not_in_merged:
    print(county)

counties_not_in_emissions = set(emissions_df['State and County']) - set(population_df['State and County'])
print("Counties in emissions but not in population:")
for county in counties_not_in_emissions:
    print(county)

emissions_df.set_index("State and County")
population_df.set_index("State and County")

# Merging the datasets

df = pd.merge(population_df,emissions_df)

df.describe()
df.columns
df.head(10)

df.set_index("State and County")

print(df.loc[df['County'].str.contains('Vald')])

print(merged_df.loc(merged_df['County'].str.contains('Vald')))
print(population_df.loc[population_df['County'].str.contains('Vald')])
print(emissions_df.loc[emissions_df['County'].str.contains('Vald')])
print(df.loc[df['County'].str.contains('Vald')])

# Exporting the cleaned datasets
population_df.to_csv("POPULATIONDATA-Cleaned.csv")
emissions_df.to_csv("EMISSIONSDATA-Cleaned.csv")
df.to_csv("AllData-MergedDS.csv")

#######
## Stopping place 1



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
population_df = pd.read_csv("POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("EMISSIONSDATA-Cleaned.csv")
merged_df = pd.read_csv("AllData-MergedDS.csv")

merged_df.set_index("State and County")
population_df.set_index("State and County")
emissions_df.set_index("State and County")

merged_df.head(10)
print(merged_df.loc[merged_df['County'].str.contains('Vald')])

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


## Cleaning the GDP dataset
gdp_df.loc[gdp_df['GeoName'].str.contains(r' \(Independent City\)', regex=True), 'GeoName'] = \
    gdp_df['GeoName'].str.replace(r' \(Independent City\)', '', regex=True).str.strip().apply(lambda x: "City of " + x)
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('St.', 'St')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('East', 'E')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('Census Area', '')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('City and Borough', '')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('Borough', '')

#Doing some checks on specific values
print(gdp_df.loc[gdp_df['GeoName'].str.contains('Emporia'),'GeoName'])
print(merged_df.loc[merged_df['County'].str.contains('Columbia'),'County'])
print(gdp_df.loc[~gdp_df['GeoName'].str.contains(','),'GeoName'])
print(merged_df.loc[merged_df['County'].str.contains('Rocky Mountain'),'County'])

#Setting Index of gdp df
gdp_df.set_index("GeoName")

gdp_df.columns

## Pivoting the GDP dataset

pivoted_df = gdp_df.pivot_table(index=['GeoName', 'GeoFIPS'], columns='Unit', values='GDP(Thousands)', aggfunc='first')

pivoted_df.head()


# Flatten the column MultiIndex, if necessary
pivoted_df.columns = [col for col in pivoted_df.columns]
# Reset the index to turn GeoName back into a column
pivoted_df.reset_index(inplace=True)
print(pivoted_df)


# Separate GeoName into County and State columns
pivoted_df[['County', 'State']] = pivoted_df['GeoName'].str.split(', ', expand=True).iloc[:, [0, 1]]
pivoted_df['County'] = pivoted_df['County'].str.rstrip()
pivoted_df['State'] = pivoted_df['State'].str.rstrip()

pivoted_df.head(10)
pivoted_df.shape

pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']
pivoted_df.set_index("State and County")


pivoted_df.sort_index()

pivoted_df.set_index("State and County",inplace=True)
merged_df.columns
pivoted_df.index

# looking at merged and pivoted dfs.
merged_df[['County','State']].head(10)
pivoted_df[['County','State']].head(10)

counties_not_in_merged = set(merged_df['State and County']) - set(pivoted_df['State and County'])
print("Counties in emissions but not in population:")
for county in counties_not_in_emissions:
    print(county)

counties_not_in_pivoted = set(pivoted_df['State and County']) - set(merged_df['State and County'])
print("Counties in emissions but not in population:")
for county in counties_not_in_emissions:
    print(county)

for state in merged_df['State'].unique():
    print("State: ", state)
    print(len(merged_df[merged_df['State']==state]))

print(pivoted_df.loc[pivoted_df['State'].isna()])
print(pivoted_df.loc[pivoted_df['County'].str.contains('District of Columbia')])
print(merged_df.loc[merged_df['County'].str.contains('District of Columbia')])

# Removing NA rows and doing more specific cleaning
pivoted_df = pivoted_df[~(pivoted_df['State'].isna())]

print(merged_df.loc[merged_df['County'].str.contains('Valdez')])
pivoted_df.loc[pivoted_df['County'].str.startswith('West '), 'County'] = pivoted_df['County'].str.replace('West', 'W')

# Removing the spaces from the end of counties who end in spaces.
for county in merged_df['County']:
    if county.endswith(' '):
        print(county)
        merged_df['County'] = merged_df['County'].str.rstrip()

#######CANNOT RUN YET, NEED TO ELIM NAS IN PIVOTED_DF
# Printing rows with NA values in pivoted_df
#pivoted_df = pivoted_df.dropna()
counties_with_asterisk = pivoted_df[pivoted_df['State'].str.contains('\*')][['County', 'State']]
print(counties_with_asterisk)

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
