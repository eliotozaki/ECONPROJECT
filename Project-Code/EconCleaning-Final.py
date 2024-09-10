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

merged_df.set_index("State and County", inplace=True)
population_df.set_index("State and County", inplace=True)
emissions_df.set_index("State and County", inplace=True)

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

pivoted_df.sort_values(by='State and County', inplace=True)

merged_df.columns
pivoted_df.index
merged_df.reset_index(inplace=True)

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
print(merged_df.loc[merged_df['County'].str.contains('Ketchikan')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Ketchikan')])
pivoted_df = pivoted_df[pivoted_df['County'] != 'Prince of Wales-Outer Ketchikan']
edited_counties = edited_counties[edited_counties['County'] != 'Prince of Wales-Outer Ketchikan']
counties_not_in_merged = counties_not_in_merged[counties_not_in_merged != ('Prince of Wales-Outer Ketchikan')]
similar_counties = {county: similar_counties[county] for county in similar_counties if county != 'Prince of Wales-Outer Ketchikan'}

## Further cleaning

#- Dealing with Valdez-Cordova, by combining the values of Chugach and Copper River
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Wrangell-Petersburg')]
print(merged_df.loc[merged_df['County'].str.contains('Chugach'),'County'])
print(merged_df.loc[merged_df['County'].str.contains('River'),['County','State']])
print(pivoted_df.loc[pivoted_df['County'].str.contains('River')])

print(merged_df.loc[merged_df['State and County'].str.contains('Vald') ])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Vald')])

pivoted_df[['Quantity index', 'Thousands of chained 2012 dollars', 'Thousands of dollars']] = pivoted_df[['Quantity index', 'Thousands of chained 2012 dollars', 'Thousands of dollars']].replace('(NA)', 0)

colnames = ['Quantity index', 'Thousands of chained 2012 dollars', 'Thousands of dollars']
vald_counties = ['Chugach','Copper River']
for c in vald_counties:
    for col in colnames:
        pivoted_df.loc[pivoted_df['County'].str.contains('Valdez-Cordova'), col] = int(pivoted_df.loc[pivoted_df['County'].str.contains('Valdez-Cordova'), col]) + int(pivoted_df.loc[pivoted_df['County'].str.contains(c), col])

print(type(pivoted_df.loc[pivoted_df['County'].str.contains('Valdez-Cordova'), 'Quantity index']))
print(pivoted_df.loc[pivoted_df['County'].str.contains('Valdez-Cordova')])

pivoted_df = pivoted_df[~(pivoted_df['County'].isin(vald_counties))]

# Finding the gdp of Valdez-Cordova in chained 2012 dollars
# - Found by averaging the chained dollar index of other counties in Alaska from 2021 and dividing GDP by it.
pivoted_df.loc[pivoted_df['County'] == 'Valdez-Cordova', 'Thousands of chained 2012 dollars'] = 1783407

##
# Changing wrangell to Wrangell City and Borough in pivoted df to match merged df.
print(merged_df[merged_df['County'].str.contains('Wrangell')])
print(pivoted_df[pivoted_df['County'].str.contains('Wrangell')])
pivoted_df.loc[pivoted_df['County'] == 'Wrangell', 'County'] = 'Wrangell City and Borough'
edited_counties.loc[edited_counties['County'] == 'Wrangell', 'County'] = 'Wrangell City and Borough'

# Updating the 'State and County' column in merged_df
merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']

# Updating the 'State and County' column in pivoted_df
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']

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
edited_counties.sort_index()

# Using that list, finding counties that are in the merged_df/not in merged df
ast_counties_not_in_merged = []
for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        ast_counties_not_in_merged.append(county)
print(ast_counties_not_in_merged)

ast_counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(ast_counties_in_merged)
#############################################################################################

combined_counties = pivoted_df.loc[pivoted_df['State'].str.endswith('*'), ['County', 'State']]
print(combined_counties)

pivoted_df = pivoted_df.sort_index()
pivoted_df.head()

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



##########
## Also doing Roanoke + Salem because of Roanoke City

print(merged_df.loc[merged_df['County'].str.contains('Roanoke')])
print(merged_df.loc[merged_df['County'].str.contains('Salem')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Roanoke')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Salem')])


roanoke_values = merged_df.loc[merged_df['County'].str.contains('Roanoke') & ~(merged_df['State'].str.contains('City of'))].copy()
salem_values = merged_df.loc[merged_df['County'].str.contains('Salem') & merged_df['State'].str.contains('VA')].copy()

# Ensure that the columns to be combined are numeric
colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# Combine the values from Southampton into Franklin
for col in colnames:
    roanoke_values[col] = roanoke_values[col].sum() + salem_values[col].sum()

# Update the Franklin row with the combined values
merged_df.loc[merged_df['County'].str.contains('Roanoke')& ~(merged_df['County'].str.contains('City of')), colnames] = roanoke_values[colnames]

# Remove the Southampton row from merged_df
merged_df = merged_df[~merged_df['County'].str.contains('City of Salem')]

# Rename the Franklin row to reflect the combined counties
merged_df.loc[merged_df['County'].str.contains('Roanoke')& ~(merged_df['County'].str.contains('City of')), 'County'] = 'Roanoke + Salem'

# Check the final merged_df
print(merged_df.loc[merged_df['County'].str.contains('Roanoke')])
print(merged_df.loc[merged_df['County'].str.contains('Salem')])




##########
## Also doing Southampton + Franklin VA because of confusion with Franklin City

print(merged_df.loc[merged_df['County'].str.contains('Franklin')])
print(merged_df.loc[merged_df['County'].str.contains('Southampton')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Franklin')])


southampton_values = merged_df.loc[merged_df['County'].str.contains('Southampton') & merged_df['State'].str.contains('VA')].copy()
franklin_values = merged_df.loc[merged_df['County'].str.contains('City of Franklin') & merged_df['State'].str.contains('VA')].copy()

# Ensure that the columns to be combined are numeric
colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# Combine the values from Southampton into Franklin
for col in colnames:
    franklin_values[col] = franklin_values[col].sum() + southampton_values[col].sum()

# Update the Franklin row with the combined values
merged_df.loc[merged_df['County'].str.contains('City of Franklin'), colnames] = franklin_values[colnames]

# Remove the Southampton row from merged_df
merged_df = merged_df[~merged_df['County'].str.contains('Southampton')]

# Rename the Franklin row to reflect the combined counties
merged_df.loc[merged_df['County'].str.contains('City of Franklin'), 'County'] = 'Southampton + Franklin'

# Check the final merged_df
print(merged_df.loc[merged_df['County'].str.contains('Southampton')])
print(merged_df.loc[merged_df['County'].str.contains('Franklin')])

############
## Additionally doing Frederick + Winchester VA because of confusion with Fredericksburg

print(merged_df.loc[merged_df['County'].str.contains('Winchester')])
print(merged_df.loc[merged_df['County'].str.contains('Frederick')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Winchester')])


win_vals = merged_df.loc[merged_df['County'].str.contains('Winchester')].copy()
fred_vals = merged_df.loc[merged_df['County'].str.contains('Frederick')& merged_df['State'].str.contains('VA') & ~(merged_df['County'].str.contains('City of'))].copy()

# Ensure that the columns to be combined are numeric
colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# Combine the values from Winchester into Frederick
for col in colnames:
    fred_vals[col] = win_vals[col].sum() + fred_vals[col].sum()

# Update the Frederick row with the combined values
merged_df.loc[merged_df['County'].str.contains('Frederick')& merged_df['State'].str.contains('VA')& ~(merged_df['County'].str.contains('City of')), colnames] = fred_vals[colnames]

# Remove the Winchester row from merged_df
merged_df = merged_df[~merged_df['County'].str.contains('Winchester')]

# Rename the Frederick row to reflect the combined counties
merged_df.loc[merged_df['County'].str.contains('Frederick')& merged_df['State'].str.contains('VA')& ~(merged_df['County'].str.contains('City of')), 'County'] = 'Frederick + Winchester'

print(merged_df.loc[merged_df['County'].str.contains('Frederick')])
print(merged_df.loc[merged_df['County'].str.contains('Winchester')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Winchester')])

############
## Additionally doing Spotsylvania + Fredericksburg VA because of confusion with Frederick

print(merged_df.loc[merged_df['County'].str.contains('Spotsylvania')])
print(merged_df.loc[merged_df['County'].str.contains('Fredericksburg')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Spotsylvania')])


spot_vals = merged_df.loc[merged_df['County'].str.contains('Spotsylvania')].copy()
fred_vals = merged_df.loc[merged_df['County'].str.contains('Fredericksburg')].copy()

# Ensure that the columns to be combined are numeric
colnames = [col for col in merged_df.columns if col not in ['Unnamed: 0', 'State and County', 'County', 'State', 'Population', 'FIPS']]

# Combine the values from Kalawao into Maui
for col in colnames:
    spot_vals[col] = spot_vals[col].sum() + fred_vals[col].sum()

# Update the Maui row with the combined values
merged_df.loc[merged_df['County'].str.contains('Spotsylvania'), colnames] = spot_vals[colnames]

# Remove the Kalawao row from merged_df
merged_df = merged_df[~merged_df['County'].str.contains('Fredericksburg')]

# Rename the Maui row to reflect the combined counties
merged_df.loc[merged_df['County'].str.contains('Spotsylvania'), 'County'] = 'Spotsylvania + Fredericksburg'

print(merged_df.loc[merged_df['County'].str.contains('Spotsylvania')])
print(merged_df.loc[merged_df['County'].str.contains('Fredericksburg')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Spotsylvania')])





#########################
#########################
#########################
###########################
############################

######## Now doing the rest, which are all in VA
print(combined_counties)
combined_counties.sort_values(by='State',inplace=True)
print(combined_counties)
combined_counties_new = combined_counties[2:]
print(combined_counties_new)
combined_counties_new = combined_counties_new[~combined_counties_new['County'].str.contains("Roanoke")]
combined_counties_new = combined_counties_new[~combined_counties_new['County'].str.contains("Franklin")]
combined_counties_new = combined_counties_new[~combined_counties_new['County'].str.contains("Fredericksburg")]
combined_counties_new = combined_counties_new[~combined_counties_new['County'].str.contains("Winchester")]
print(combined_counties_new)

for county in combined_counties_new['County']:
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


merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']



###########################################################################################
## Final checks for combined counties in pivoted_df that arent in merged_df

counties_not_in_gdp = set(merged_df['State and County']) - set(pivoted_df['State and County'])
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

# Editing Alutians East
pivoted_df.loc[pivoted_df['County'].str.contains('Aleutians E'), 'County'] = 'Aleutians East'


print(merged_df.columns)
print(pivoted_df['State'].unique())

# Find counties and states from pivoted_df where the state value is not in the state values of merged_df
unique_states = pivoted_df.loc[~(pivoted_df['State'].isin(merged_df['State']))& ~(pivoted_df['State'].str.contains('AK')), ['County', 'State']]
print(unique_states)
print(pivoted_df.loc[pivoted_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['County']), ['County','State']])
# All are from VA, so changing the state to VA
pivoted_df.loc[pivoted_df['State'].isin(unique_states['State']), 'State'] = 'VA'

# Checking to make sure no counties in either ds have spaces at the end of counties (mould mess with merge)
counties_with_trailing_space = merged_df.loc[merged_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])
counties_with_trailing_space = pivoted_df.loc[pivoted_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])

# Updating State and County columns
merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


#############################################################################################
## More checks


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


#############################################################################################

# Changing Anchorage to match in both datasets
print(merged_df.loc[merged_df['County'].str.contains('Anchorage')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage')])
pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage'), 'County'] = 'Anchorage'
pivoted_df.loc[pivoted_df['State and County'].str.contains('Anchorage'), 'State and County'] = 'AK Anchorage'


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

print(merged_df[['State and County', 'State', 'County']].head(20))
print(pivoted_df[['State and County', 'State', 'County']].head(25))


##### Checking more similar counties, along with specifically counties in VA
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

print(merged_df.loc[merged_df['County'].str.contains('Buena Vista')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Buena Vista')])

#############################################
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

#########
## No more counties in merged_df that arent in gdp_df
## Need to do last edits for counties:
## - CT Hartford
## - CT Litchfield
## - CT Tolland
## - IN DeKalb
## - LA LaSalle
## - MT Garfield
## - VA City of Roanoke
## - VA Franklin

# Hartford/CT Counties:
print(merged_df.loc[merged_df['County'].str.contains('Hartford')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Hartford')])
print(len(merged_df.loc[merged_df['State'].str.contains('CT')]))
print(len(pivoted_df.loc[pivoted_df['State'].str.contains('CT')]))
## might need to remove Hartford, Litchfield, Tolland from pivoted_df
## - Hartford, Litchfield, and Tolland are all major counties in CT; not likely to be merged with another county. Thus, can only 
##   assume they were left out for some reason. Thus, have to omit in pivoted

pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('CT') & pivoted_df['County'].isin(['Tolland', 'Hartford', 'Litchfield']))]

### Finishing the rest of those counties:
#LaSalle
print(merged_df.loc[merged_df['County'].str.contains('Salle')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Salle')])
print(len(merged_df.loc[merged_df['State'].str.contains('LA')]))
print(len(pivoted_df.loc[pivoted_df['State'].str.contains('LA')]))
# LaSalle not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('LA') & pivoted_df['County'].str.contains('LaSalle'))]

# DeKalb
print(merged_df.loc[merged_df['County'].str.contains('DeKalb')])
print(merged_df.loc[merged_df['State'].str.contains('IN')])
print(merged_df.loc[merged_df['County'].str.contains('De')& merged_df['State'].str.contains('IN')])
# Dekalb not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('IN') & pivoted_df['County'].str.contains('DeKalb'))]

# Garfield
print(merged_df.loc[merged_df['County'].str.contains('Garfield')])
print(merged_df.loc[merged_df['State'].str.contains('MT')])
print(merged_df.loc[merged_df['County'].str.contains('Garfield')& merged_df['State'].str.contains('MT')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Garfield')])
# Garfield not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('MT') & pivoted_df['County'].str.contains('Garfield'))]


print(merged_df[merged_df.isna().any(axis=1)])
print(pivoted_df[pivoted_df.isna().any(axis=1)])
print(merged_df.loc[merged_df['County'].str.contains('NA')])


merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']





pivoted_df.sort_values(by='State and County', inplace=True) 
merged_df.sort_values(by='State and County', inplace=True)




##############################################################################
## FINAL CHECK
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

pivoted_df.columns

gdp_merged = pd.merge(merged_df, pivoted_df, on='State and County', how='outer')

print(len(merged_df))
print(len(pivoted_df))
print(len(gdp_merged))
gdp_merged.columns

gdp_merged.to_csv('gdp_merged.csv', index=False)

#######################
# New Stopping place


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap

merged_df = pd.read_csv('gdp_merged.csv')

gdp_df = pd.read_csv("Cleaned-Datasets\CAGDP1__ALL_AREAS_2001_2021.csv", encoding='windows-1252')
gdp_df.columns


# Initial look at the dataset
gdp_df.describe()
gdp_df.head(10)

gdp_df = gdp_df[:-4]
print(gdp_df.tail(10))

## Inital Cleaning

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
gdp_df.tail(20)

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


#Setting Index of gdp df
gdp_df.set_index("GeoName")

## Pivoting the GDP dataset

pivoted_df = gdp_df.pivot_table(index=['GeoName', 'GeoFIPS'], columns='Unit', values='2021', aggfunc='first')
pivoted_df.head()
pivoted_df.sort_values(by='GeoName', inplace=True)

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

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']

pivoted_df.sort_values(by='State and County', inplace=True)

# looking at merged and pivoted dfs.
pivoted_df[['County','State']].head(10)

# Removing NA rows and doing more specific cleaning
pivoted_df = pivoted_df[~(pivoted_df['State'].isna())]

pivoted_df.loc[pivoted_df['County'].str.startswith('West '), 'County'] = pivoted_df['County'].str.replace('West', 'W')

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']

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



## Finding rows in pivoted_df with asterisks in the 'State' column
print(pivoted_df[pivoted_df['State'].str.contains('\*')])
edited_counties = pivoted_df[pivoted_df['State'].str.contains('\*')]
edited_counties.head(len(edited_counties))
edited_counties.sort_index()

# Using that list, finding counties that are in the merged_df/not in merged df
ast_counties_not_in_merged = []
for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        ast_counties_not_in_merged.append(county)
print(ast_counties_not_in_merged)

ast_counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(ast_counties_in_merged)
#############################################################################################

combined_counties = pivoted_df.loc[pivoted_df['State'].str.endswith('*'), ['County', 'State']]
print(combined_counties)

pivoted_df = pivoted_df.sort_index()
pivoted_df.head()

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']



###########################################################################################
## Final checks for combined counties in pivoted_df that arent in merged_df

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

# Editing Alutians East
pivoted_df.loc[pivoted_df['County'].str.contains('Aleutians E'), 'County'] = 'Aleutians East'


print(merged_df.columns)
print(pivoted_df['State'].unique())

# Find counties and states from pivoted_df where the state value is not in the state values of merged_df
unique_states = pivoted_df.loc[~(pivoted_df['State'].isin(merged_df['State']))& ~(pivoted_df['State'].str.contains('AK')), ['County', 'State']]
print(unique_states)
print(pivoted_df.loc[pivoted_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['County']), ['County','State']])
# All are from VA, so changing the state to VA
pivoted_df.loc[pivoted_df['State'].isin(unique_states['State']), 'State'] = 'VA'

# Checking to make sure no counties in either ds have spaces at the end of counties (mould mess with merge)
counties_with_trailing_space = merged_df.loc[merged_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])
counties_with_trailing_space = pivoted_df.loc[pivoted_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])

# Updating State and County columns
merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


#############################################################################################
## More checks


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


#############################################################################################

# Changing Anchorage to match in both datasets
print(merged_df.loc[merged_df['County'].str.contains('Anchorage')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage')])
pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage'), 'County'] = 'Anchorage'
pivoted_df.loc[pivoted_df['State and County'].str.contains('Anchorage'), 'State and County'] = 'AK Anchorage'


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

print(merged_df[['State and County', 'State', 'County']].head(20))
print(pivoted_df[['State and County', 'State', 'County']].head(25))


##### Checking more similar counties, along with specifically counties in VA
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

print(merged_df.loc[merged_df['County'].str.contains('Buena Vista')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Buena Vista')])

#############################################
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

#########
## No more counties in merged_df that arent in gdp_df
## Need to do last edits for counties:
## - CT Hartford
## - CT Litchfield
## - CT Tolland
## - IN DeKalb
## - LA LaSalle
## - MT Garfield
## - VA City of Roanoke
## - VA Franklin

# Hartford/CT Counties:
print(merged_df.loc[merged_df['County'].str.contains('Hartford')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Hartford')])
print(len(merged_df.loc[merged_df['State'].str.contains('CT')]))
print(len(pivoted_df.loc[pivoted_df['State'].str.contains('CT')]))
## might need to remove Hartford, Litchfield, Tolland from pivoted_df
## - Hartford, Litchfield, and Tolland are all major counties in CT; not likely to be merged with another county. Thus, can only 
##   assume they were left out for some reason. Thus, have to omit in pivoted

pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('CT') & pivoted_df['County'].isin(['Tolland', 'Hartford', 'Litchfield']))]

### Finishing the rest of those counties:
#LaSalle
print(merged_df.loc[merged_df['County'].str.contains('Salle')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Salle')])
print(len(merged_df.loc[merged_df['State'].str.contains('LA')]))
print(len(pivoted_df.loc[pivoted_df['State'].str.contains('LA')]))
# LaSalle not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('LA') & pivoted_df['County'].str.contains('LaSalle'))]

# DeKalb
print(merged_df.loc[merged_df['County'].str.contains('DeKalb')])
print(merged_df.loc[merged_df['State'].str.contains('IN')])
print(merged_df.loc[merged_df['County'].str.contains('De')& merged_df['State'].str.contains('IN')])
# Dekalb not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('IN') & pivoted_df['County'].str.contains('DeKalb'))]

# Garfield
print(merged_df.loc[merged_df['County'].str.contains('Garfield')])
print(merged_df.loc[merged_df['State'].str.contains('MT')])
print(merged_df.loc[merged_df['County'].str.contains('Garfield')& merged_df['State'].str.contains('MT')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Garfield')])
# Garfield not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('MT') & pivoted_df['County'].str.contains('Garfield'))]


print(merged_df[merged_df.isna().any(axis=1)])
print(pivoted_df[pivoted_df.isna().any(axis=1)])
print(merged_df.loc[merged_df['County'].str.contains('NA')])

pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Wrangell-Petersburg')]


# Find counties and states from pivoted_df where the state value is not in the state values of merged_df
unique_states = pivoted_df.loc[~(pivoted_df['State'].isin(merged_df['State and County'].str[:2]))& ~(pivoted_df['State'].str.contains('AK')), ['County', 'State']]
print(unique_states)
print(pivoted_df.loc[pivoted_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['County']), ['County','State']])
# All are from VA, so changing the state to VA
pivoted_df.loc[pivoted_df['State'].isin(unique_states['State']), 'State'] = 'VA'

# Wrangell
pivoted_df.loc[pivoted_df['County'].str.contains('Wrangell'), 'County'] = 'Wrangell City and Borough'


#- Dealing with Valdez-Cordova, by combining the values of Chugach and Copper River
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Wrangell-Petersburg')]
print(merged_df.loc[merged_df['County'].str.contains('Chugach'),'County'])
print(merged_df.loc[merged_df['County'].str.contains('River'),['County','State']])
print(pivoted_df.loc[pivoted_df['County'].str.contains('River')])

print(merged_df.loc[merged_df['State and County'].str.contains('Vald') ])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Vald')])

pivoted_df[['Quantity index', 'Thousands of chained 2012 dollars', 'Thousands of dollars']] = pivoted_df[['Quantity index', 'Thousands of chained 2012 dollars', 'Thousands of dollars']].replace('(NA)', 0)

colnames = ['Quantity index', 'Thousands of chained 2012 dollars', 'Thousands of dollars']
vald_counties = ['Chugach','Copper River']
for c in vald_counties:
    for col in colnames:
        pivoted_df.loc[pivoted_df['County'].str.contains('Valdez-Cordova'), col] = int(pivoted_df.loc[pivoted_df['County'].str.contains('Valdez-Cordova'), col]) + int(pivoted_df.loc[pivoted_df['County'].str.contains(c), col])

print(type(pivoted_df.loc[pivoted_df['County'].str.contains('Valdez-Cordova'), 'Quantity index']))
print(pivoted_df.loc[pivoted_df['County'].str.contains('Valdez-Cordova')])

pivoted_df = pivoted_df[~(pivoted_df['County'].isin(vald_counties))]

# Finding the gdp of Valdez-Cordova in chained 2012 dollars
# - Found by averaging the chained dollar index of other counties in Alaska from 2021 and dividing GDP by it.
pivoted_df.loc[pivoted_df['County'] == 'Valdez-Cordova', 'Thousands of chained 2012 dollars'] = 1783407

# Dropping Prince of Wales-Outer Ketchikan and Skagway-Hoonah-Angoon
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Prince of Wales-Outer Ketchikan')]
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Skagway-Hoonah-Angoon')]


pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


pivoted_df.sort_values(by='State and County', inplace=True) 



##############################################################################
## FINAL CHECK
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

print(len(pivoted_df))

merged_df = pd.merge(merged_df, pivoted_df, on='State and County', how='outer')
print(len(merged_df))
print(merged_df.isna().sum())

################################################################################
## Running above code on all government income datasets
merged_df = pd.read_csv('gdp_merged.csv')
#datasets = []
df = pd.read_csv("Cleaned-Datasets/CAGDP11__ALL_AREAS_2002_2021.csv", encoding='windows-1252', low_memory=False)
df.drop(columns=['Region'],inplace=True)
#datasets.append(df)
#df = pd.read_csv("Cleaned-Datasets/CAGDP2__ALL_AREAS_2001_2021.csv", encoding='windows-1252')
#df.drop(columns=['2001','Region'], inplace=True)
#datasets.append(df)

gdp_df = df
gdp_df.tail(20)
gdp_df = gdp_df.drop(columns=['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
gdp_df = gdp_df[:-4]



## Inital Cleaning

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
gdp_df.tail(20)
gdp_df.head(20)

# Remove rows with Geoname of United States and any aggregated state rows.
gdp_df = gdp_df[~gdp_df['GeoName'].isin(states + ['United States *', 'Far West' , 'Great Lakes', 'Mideast', 'New England', 'Plains', 'Rocky Mountain', 'Southeast', 'Southwest'])]

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


#Setting Index of gdp df
gdp_df.set_index("GeoName")

## Pivoting the GDP dataset

pivoted_df = gdp_df.pivot_table(index=['GeoName', 'GeoFIPS'], columns='Description', values='2021', aggfunc='first')
pivoted_df.head()
pivoted_df.sort_values(by='GeoName', inplace=True)

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

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']

pivoted_df.sort_values(by='State and County', inplace=True)

# looking at merged and pivoted dfs.
pivoted_df[['County','State']].head(10)

# Removing NA rows and doing more specific cleaning
pivoted_df = pivoted_df[~(pivoted_df['State'].isna())]

pivoted_df.loc[pivoted_df['County'].str.startswith('West '), 'County'] = pivoted_df['County'].str.replace('West', 'W')

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']

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



## Finding rows in pivoted_df with asterisks in the 'State' column
print(pivoted_df[pivoted_df['State'].str.contains('\*')])
edited_counties = pivoted_df[pivoted_df['State'].str.contains('\*')]
edited_counties.head(len(edited_counties))
edited_counties.sort_index()

# Using that list, finding counties that are in the merged_df/not in merged df
ast_counties_not_in_merged = []
for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        ast_counties_not_in_merged.append(county)
print(ast_counties_not_in_merged)

ast_counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(ast_counties_in_merged)
#############################################################################################

combined_counties = pivoted_df.loc[pivoted_df['State'].str.endswith('*'), ['County', 'State']]
print(combined_counties)

pivoted_df = pivoted_df.sort_index()
pivoted_df.head()

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']



###########################################################################################
## Final checks for combined counties in pivoted_df that arent in merged_df

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

# Editing Alutians East
pivoted_df.loc[pivoted_df['County'].str.contains('Aleutians E'), 'County'] = 'Aleutians East'


print(merged_df.columns)
print(pivoted_df['State'].unique())

# Find counties and states from pivoted_df where the state value is not in the state values of merged_df
unique_states = pivoted_df.loc[~(pivoted_df['State'].isin(merged_df['State']))& ~(pivoted_df['State'].str.contains('AK')), ['County', 'State']]
print(unique_states)
print(pivoted_df.loc[pivoted_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['County']), ['County','State']])
# All are from VA, so changing the state to VA
pivoted_df.loc[pivoted_df['State'].isin(unique_states['State']), 'State'] = 'VA'

# Checking to make sure no counties in either ds have spaces at the end of counties (mould mess with merge)
counties_with_trailing_space = merged_df.loc[merged_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])
counties_with_trailing_space = pivoted_df.loc[pivoted_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])

# Updating State and County columns
merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


#############################################################################################
## More checks


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


#############################################################################################

# Changing Anchorage to match in both datasets
print(merged_df.loc[merged_df['County'].str.contains('Anchorage')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage')])
pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage'), 'County'] = 'Anchorage'
pivoted_df.loc[pivoted_df['State and County'].str.contains('Anchorage'), 'State and County'] = 'AK Anchorage'


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

print(merged_df[['State and County', 'State', 'County']].head(20))
print(pivoted_df[['State and County', 'State', 'County']].head(25))


##### Checking more similar counties, along with specifically counties in VA
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

print(merged_df.loc[merged_df['County'].str.contains('Buena Vista')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Buena Vista')])

#############################################
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

#########
## No more counties in merged_df that arent in gdp_df
## Need to do last edits for counties:
## - CT Hartford
## - CT Litchfield
## - CT Tolland
## - IN DeKalb
## - LA LaSalle
## - MT Garfield
## - VA City of Roanoke
## - VA Franklin

# Hartford/CT Counties:
print(merged_df.loc[merged_df['County'].str.contains('Hartford')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Hartford')])
print(len(merged_df.loc[merged_df['State'].str.contains('CT')]))
print(len(pivoted_df.loc[pivoted_df['State'].str.contains('CT')]))
## might need to remove Hartford, Litchfield, Tolland from pivoted_df
## - Hartford, Litchfield, and Tolland are all major counties in CT; not likely to be merged with another county. Thus, can only 
##   assume they were left out for some reason. Thus, have to omit in pivoted

pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('CT') & pivoted_df['County'].isin(['Tolland', 'Hartford', 'Litchfield']))]

### Finishing the rest of those counties:
#LaSalle
print(merged_df.loc[merged_df['County'].str.contains('Salle')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Salle')])
print(len(merged_df.loc[merged_df['State'].str.contains('LA')]))
print(len(pivoted_df.loc[pivoted_df['State'].str.contains('LA')]))
# LaSalle not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('LA') & pivoted_df['County'].str.contains('LaSalle'))]

# DeKalb
print(merged_df.loc[merged_df['County'].str.contains('DeKalb')])
print(merged_df.loc[merged_df['State'].str.contains('IN')])
print(merged_df.loc[merged_df['County'].str.contains('De')& merged_df['State'].str.contains('IN')])
# Dekalb not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('IN') & pivoted_df['County'].str.contains('DeKalb'))]

# Garfield
print(merged_df.loc[merged_df['County'].str.contains('Garfield')])
print(merged_df.loc[merged_df['State'].str.contains('MT')])
print(merged_df.loc[merged_df['County'].str.contains('Garfield')& merged_df['State'].str.contains('MT')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Garfield')])
# Garfield not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('MT') & pivoted_df['County'].str.contains('Garfield'))]


print(merged_df[merged_df.isna().any(axis=1)])
print(pivoted_df[pivoted_df.isna().any(axis=1)])
print(merged_df.loc[merged_df['County'].str.contains('NA')])

pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Wrangell-Petersburg')]


# Find counties and states from pivoted_df where the state value is not in the state values of merged_df
unique_states = pivoted_df.loc[~(pivoted_df['State'].isin(merged_df['State and County'].str[:2]))& ~(pivoted_df['State'].str.contains('AK')), ['County', 'State']]
print(unique_states)
print(pivoted_df.loc[pivoted_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['County']), ['County','State']])
# All are from VA, so changing the state to VA
pivoted_df.loc[pivoted_df['State'].isin(unique_states['State']), 'State'] = 'VA'

# Wrangell
pivoted_df.loc[pivoted_df['County'].str.contains('Wrangell'), 'County'] = 'Wrangell City and Borough'


#- Dealing with Valdez-Cordova, by combining the values of Chugach and Copper River
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Wrangell-Petersburg')]
print(merged_df.loc[merged_df['County'].str.contains('Chugach'),'County'])
print(merged_df.loc[merged_df['County'].str.contains('River'),['County','State']])
print(pivoted_df.loc[pivoted_df['County'].str.contains('River')])

print(merged_df.loc[merged_df['State and County'].str.contains('Vald') ])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Vald')])


vald_counties = ['Chugach','Copper River']

pivoted_df.replace('(NA)', 0, inplace=True)

print(pivoted_df)

pivoted_df = pivoted_df[~(pivoted_df['County'].isin(vald_counties))]
# Define a function to replace '(D)' with the mean of the column
def replace_d_with_mean(col):
    # Convert the column to numeric, coercing errors for non-numeric values
    col_numeric = pd.to_numeric(col, errors='coerce')
    
    # Calculate the mean of the column, ignoring NaN values
    col_mean = col_numeric.mean()
    
    # Replace '(D)' with the column mean
    return col.replace('(D)', col_mean)

# Apply the function to all columns in the dataframe
pivoted_df = pivoted_df.apply(replace_d_with_mean)

# Finding the gdp of Valdez-Cordova in chained 2012 dollars
# - Found by averaging the chained dollar index of other counties in Alaska from 2021 and dividing GDP by it.

# Dropping Prince of Wales-Outer Ketchikan and Skagway-Hoonah-Angoon
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Prince of Wales-Outer Ketchikan')]
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Skagway-Hoonah-Angoon')]


pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


pivoted_df.sort_values(by='State and County', inplace=True) 



##############################################################################
## FINAL CHECK
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

print(len(pivoted_df))

merged_df = pd.merge(merged_df, pivoted_df, on='State and County', how='outer')
merged_df.to_csv('gdp_merged.csv', index=False)





################################
## Cleaning CAGDP2


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap


merged_df = pd.read_csv('gdp_merged.csv')
#datasets = []
df = pd.read_csv("Cleaned-Datasets/CAGDP2__ALL_AREAS_2001_2021.csv", encoding='windows-1252', low_memory=False)
df.drop(columns=['Region','2001'],inplace=True)
#datasets.append(df)
#df = pd.read_csv("Cleaned-Datasets/CAGDP2__ALL_AREAS_2001_2021.csv", encoding='windows-1252')
#df.drop(columns=['2001','Region'], inplace=True)
#datasets.append(df)

gdp_df = df
gdp_df.tail(20)
gdp_df = gdp_df.drop(columns=['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
gdp_df = gdp_df[:-4]



## Inital Cleaning

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
gdp_df.tail(20)
gdp_df.head(20)

# Remove rows with Geoname of United States and any aggregated state rows.
gdp_df = gdp_df[~gdp_df['GeoName'].isin(states + ['United States *', 'Far West' , 'Great Lakes', 'Mideast', 'New England', 'Plains', 'Rocky Mountain', 'Southeast', 'Southwest'])]

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


#Setting Index of gdp df
gdp_df.set_index("GeoName")

## Pivoting the GDP dataset

pivoted_df = gdp_df.pivot_table(index=['GeoName', 'GeoFIPS'], columns='Description', values='2021', aggfunc='first')
pivoted_df.head()
pivoted_df.sort_values(by='GeoName', inplace=True)

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

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']

pivoted_df.sort_values(by='State and County', inplace=True)

# looking at merged and pivoted dfs.
pivoted_df[['County','State']].head(10)

# Removing NA rows and doing more specific cleaning
pivoted_df = pivoted_df[~(pivoted_df['State'].isna())]

pivoted_df.loc[pivoted_df['County'].str.startswith('West '), 'County'] = pivoted_df['County'].str.replace('West', 'W')

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']

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



## Finding rows in pivoted_df with asterisks in the 'State' column
print(pivoted_df[pivoted_df['State'].str.contains('\*')])
edited_counties = pivoted_df[pivoted_df['State'].str.contains('\*')]
edited_counties.head(len(edited_counties))
edited_counties.sort_index()

# Using that list, finding counties that are in the merged_df/not in merged df
ast_counties_not_in_merged = []
for county, state in zip(edited_counties['County'], edited_counties['State'].str[:2]):
    if (county, state) not in zip(merged_df['County'], merged_df['State']):
        ast_counties_not_in_merged.append(county)
print(ast_counties_not_in_merged)

ast_counties_in_merged = list(set(edited_counties['County']) - set(counties_not_in_merged))
print(ast_counties_in_merged)
#############################################################################################

combined_counties = pivoted_df.loc[pivoted_df['State'].str.endswith('*'), ['County', 'State']]
print(combined_counties)

pivoted_df = pivoted_df.sort_index()
pivoted_df.head()

pivoted_df['State and County'] = pivoted_df['State'].str[:2] + ' ' + pivoted_df['County']



###########################################################################################
## Final checks for combined counties in pivoted_df that arent in merged_df

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

# Editing Alutians East
pivoted_df.loc[pivoted_df['County'].str.contains('Aleutians E'), 'County'] = 'Aleutians East'


print(merged_df.columns)
print(pivoted_df['State'].unique())

# Find counties and states from pivoted_df where the state value is not in the state values of merged_df
unique_states = pivoted_df.loc[~(pivoted_df['State'].isin(merged_df['State']))& ~(pivoted_df['State'].str.contains('AK')), ['County', 'State']]
print(unique_states)
print(pivoted_df.loc[pivoted_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['County']), ['County','State']])
# All are from VA, so changing the state to VA
pivoted_df.loc[pivoted_df['State'].isin(unique_states['State']), 'State'] = 'VA'

# Checking to make sure no counties in either ds have spaces at the end of counties (mould mess with merge)
counties_with_trailing_space = merged_df.loc[merged_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])
counties_with_trailing_space = pivoted_df.loc[pivoted_df['County'].str.endswith(' ')]
print(counties_with_trailing_space['County'])

# Updating State and County columns
merged_df['State and County'] = merged_df['State'] + ' ' + merged_df['County']
pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


#############################################################################################
## More checks


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


#############################################################################################

# Changing Anchorage to match in both datasets
print(merged_df.loc[merged_df['County'].str.contains('Anchorage')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage')])
pivoted_df.loc[pivoted_df['County'].str.contains('Anchorage'), 'County'] = 'Anchorage'
pivoted_df.loc[pivoted_df['State and County'].str.contains('Anchorage'), 'State and County'] = 'AK Anchorage'


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

print(merged_df[['State and County', 'State', 'County']].head(20))
print(pivoted_df[['State and County', 'State', 'County']].head(25))


##### Checking more similar counties, along with specifically counties in VA
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

print(merged_df.loc[merged_df['County'].str.contains('Buena Vista')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Buena Vista')])

#############################################
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

#########
## No more counties in merged_df that arent in gdp_df
## Need to do last edits for counties:
## - CT Hartford
## - CT Litchfield
## - CT Tolland
## - IN DeKalb
## - LA LaSalle
## - MT Garfield
## - VA City of Roanoke
## - VA Franklin

# Hartford/CT Counties:
print(merged_df.loc[merged_df['County'].str.contains('Hartford')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Hartford')])
print(len(merged_df.loc[merged_df['State'].str.contains('CT')]))
print(len(pivoted_df.loc[pivoted_df['State'].str.contains('CT')]))
## might need to remove Hartford, Litchfield, Tolland from pivoted_df
## - Hartford, Litchfield, and Tolland are all major counties in CT; not likely to be merged with another county. Thus, can only 
##   assume they were left out for some reason. Thus, have to omit in pivoted

pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('CT') & pivoted_df['County'].isin(['Tolland', 'Hartford', 'Litchfield']))]

### Finishing the rest of those counties:
#LaSalle
print(merged_df.loc[merged_df['County'].str.contains('Salle')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Salle')])
print(len(merged_df.loc[merged_df['State'].str.contains('LA')]))
print(len(pivoted_df.loc[pivoted_df['State'].str.contains('LA')]))
# LaSalle not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('LA') & pivoted_df['County'].str.contains('LaSalle'))]

# DeKalb
print(merged_df.loc[merged_df['County'].str.contains('DeKalb')])
print(merged_df.loc[merged_df['State'].str.contains('IN')])
print(merged_df.loc[merged_df['County'].str.contains('De')& merged_df['State'].str.contains('IN')])
# Dekalb not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('IN') & pivoted_df['County'].str.contains('DeKalb'))]

# Garfield
print(merged_df.loc[merged_df['County'].str.contains('Garfield')])
print(merged_df.loc[merged_df['State'].str.contains('MT')])
print(merged_df.loc[merged_df['County'].str.contains('Garfield')& merged_df['State'].str.contains('MT')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Garfield')])
# Garfield not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('MT') & pivoted_df['County'].str.contains('Garfield'))]


print(merged_df[merged_df.isna().any(axis=1)])
print(pivoted_df[pivoted_df.isna().any(axis=1)])
print(merged_df.loc[merged_df['County'].str.contains('NA')])

pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Wrangell-Petersburg')]


# Find counties and states from pivoted_df where the state value is not in the state values of merged_df
unique_states = pivoted_df.loc[~(pivoted_df['State'].isin(merged_df['State and County'].str[:2]))& ~(pivoted_df['State'].str.contains('AK')), ['County', 'State']]
print(unique_states)
print(pivoted_df.loc[pivoted_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['State'])])
print(merged_df.loc[merged_df['County'].isin(unique_states['County']), ['County','State']])
# All are from VA, so changing the state to VA
pivoted_df.loc[pivoted_df['State'].isin(unique_states['State']), 'State'] = 'VA'

# Wrangell
pivoted_df.loc[pivoted_df['County'].str.contains('Wrangell'), 'County'] = 'Wrangell City and Borough'


#- Dealing with Valdez-Cordova, by combining the values of Chugach and Copper River
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Wrangell-Petersburg')]
print(merged_df.loc[merged_df['County'].str.contains('Chugach'),'County'])
print(merged_df.loc[merged_df['County'].str.contains('River'),['County','State']])
print(pivoted_df.loc[pivoted_df['County'].str.contains('River')])

print(merged_df.loc[merged_df['State and County'].str.contains('Vald') ])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Vald')])


vald_counties = ['Chugach','Copper River']

pivoted_df.replace('(NA)', 0, inplace=True)

print(pivoted_df)

pivoted_df = pivoted_df[~(pivoted_df['County'].isin(vald_counties))]
# Define a function to replace '(D)' with the mean of the column
def replace_d_with_mean(col):
    # Convert the column to numeric, coercing errors for non-numeric values
    col_numeric = pd.to_numeric(col, errors='coerce')
    
    # Calculate the mean of the column, ignoring NaN values
    col_mean = col_numeric.mean()
    
    # Replace '(D)' with the column mean
    return col.replace('(D)', col_mean)

# Apply the function to all columns in the dataframe
pivoted_df = pivoted_df.apply(replace_d_with_mean)

# Finding the gdp of Valdez-Cordova in chained 2012 dollars
# - Found by averaging the chained dollar index of other counties in Alaska from 2021 and dividing GDP by it.

# Dropping Prince of Wales-Outer Ketchikan and Skagway-Hoonah-Angoon
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Prince of Wales-Outer Ketchikan')]
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Skagway-Hoonah-Angoon')]


pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


pivoted_df.sort_values(by='State and County', inplace=True) 



##############################################################################
## FINAL CHECK
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

print(len(pivoted_df))
print(len(pivoted_df.isna().any(axis=1)))
print(pivoted_df.isna().sum())
pivoted_df.head(50)

# Print all values in pivoted_df that contain "(NA)"
print(pivoted_df[pivoted_df.astype(str).apply(lambda x: x.str.contains('(D)')).any(axis=1)])

# Apply the function to all columns in the dataframe
pivoted_df = pivoted_df.apply(replace_d_with_mean)

# Summarize the pivoted_df dataframe
summary = pivoted_df.describe()
print(summary)
print(merged_df.columns[merged_df.columns.str.contains('County')])
merged_df.drop(columns=['County_x','State_x'], inplace=True)

merged_df = pd.merge(merged_df, pivoted_df, on='State and County', how='outer')
merged_df.to_csv('gdp_merged.csv', index=False)




################################
## Cleaning Tax Data

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap

merged_df = pd.read_csv('gdp_merged.csv')

gdp_df = pd.read_csv('Cleaned-Datasets/21incyallagi.csv', encoding='windows-1252')

gdp_df.describe()
print(len(gdp_df))


## Inital Cleaning

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
gdp_df.tail(20)
gdp_df.head(20)

# Remove rows with Geoname of United States and any aggregated state rows.
gdp_df = gdp_df[~gdp_df['COUNTYNAME'].isin(states + ['United States *', 'Far West' , 'Great Lakes', 'Mideast', 'New England', 'Plains', 'Rocky Mountain', 'Southeast', 'Southwest'])]

print(len(gdp_df)/8)
for column in gdp_df.columns:
    print(column)

gdp_df['State and County'] = gdp_df['STATE'] + ' ' + gdp_df['COUNTYNAME']
gdp_df.sort_values(by='State and County', inplace=True)
gdp_df.head(20)


# Assuming gdp_df is your DataFrame
columns_to_sum = [
    'N1', 'mars1', 'MARS2', 'MARS4', 'ELF', 'CPREP', 'PREP', 'DIR_DEP', 'VRTCRIND', 'N2', 
    'TOTAL_VITA', 'VITA', 'TCE', 'VITA_EIC', 'RAC', 'ELDERLY', 'A00100', 'N02650', 'A02650', 
    'N00200', 'A00200', 'N00300', 'A00300', 'N00600', 'A00600', 'N00650', 'A00650', 'N00700', 
    'A00700', 'N00900', 'A00900', 'N01000', 'A01000', 'N01400', 'A01400', 'N01700', 'A01700', 
    'SCHF', 'N02300', 'A02300', 'N02500', 'A02500', 'N26270', 'A26270', 'N02900', 'A02900', 
    'N03220', 'A03220', 'N03300', 'A03300', 'N03270', 'A03270', 'N03150', 'A03150', 'N03210', 
    'A03210', 'N02910', 'A02910', 'N04450', 'A04450', 'N04100', 'A04100', 'N04200', 'A04200', 
    'N04470', 'A04470', 'A00101', 'N17000', 'A17000', 'N18425', 'A18425', 'N18450', 'A18450', 
    'N18500', 'A18500', 'N18800', 'A18800', 'N18460', 'A18460', 'N18300', 'A18300', 'N19300', 
    'A19300', 'N19500', 'A19500', 'N19530', 'A19530', 'N19550', 'A19550', 'N19570', 'A19570', 
    'N19700', 'A19700', 'N20950', 'A20950', 'N04475', 'A04475', 'N04800', 'A04800', 'N05800', 
    'A05800', 'N09600', 'A09600', 'N05780', 'A05780', 'N07100', 'A07100', 'N07300', 'A07300', 
    'N07180', 'A07180', 'N07230', 'A07230', 'N07240', 'A07240', 'N07225', 'A07225', 'N07260', 
    'A07260', 'N09400', 'A09400', 'N85770', 'A85770', 'N85775', 'A85775', 'N10600', 'A10600', 
    'N59660', 'A59660', 'N59720', 'A59720', 'N11070', 'A11070', 'N10960', 'A10960', 'N11560', 
    'A11560', 'N11450', 'A11450', 'N11520', 'A11520', 'N11530', 'A11530', 'N10970', 'A10970', 
    'N10971', 'A10971', 'N06500', 'A06500', 'N10300', 'A10300', 'N85530', 'A85530', 'N85300', 
    'A85300', 'N11901', 'A11901', 'N11900', 'A11900', 'N11902', 'A11902', 'N12000', 'A12000'
]

# Group by the specified columns and sum the specified columns
pivoted_df = gdp_df.groupby(['State and County','STATEFIPS', 'STATE', 'COUNTYFIPS', 'COUNTYNAME'])[columns_to_sum].sum().reset_index()


pivoted_df.head()
print(len(pivoted_df))

pivoted_df['State and County'] = pivoted_df['State and County'].str.rstrip()

for i in range(len(' County'), 0, -1):
    pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace(f"{' County'[:i]}$", '', regex=True)

for i in range(len(' Borough'), 0, -1):
    pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace(f"{' Borough'[:i]}$", '', regex=True)

pivoted_df.head()

print(merged_df.loc[merged_df['State and County'].str.contains('Borough')])
pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Wrangell City and'), 'COUNTYNAME'] = 'Wrangell City and Borough'

print(merged_df.loc[merged_df['State and County'].str.contains('Sitka')])

pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace(" City and$", '', regex=True)

pivoted_df.head(50)

print(merged_df.loc[merged_df['State and County'].str.contains('Census Area')])

for i in range(len(' Census Area'), 1, -1):
    pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace(f"{' Census Area'[:i]}$", '', regex=True)

pivoted_df['State and County'] = pivoted_df['STATE'] + ' ' + pivoted_df['COUNTYNAME']

## Checking for counties not in merged/pivoted
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


## Checking for ds specific organizational tendencies

print(merged_df.loc[merged_df['State and County'].str.contains(' City$')])
# Changing 'city' to 'City' in pivoted_df
pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Carson city')| pivoted_df['COUNTYNAME'].str.contains('Charles city'), 'COUNTYNAME'] = pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Carson city')| pivoted_df['COUNTYNAME'].str.contains('Charles city'), 'COUNTYNAME'].str.replace('city', 'City')


print(pivoted_df.loc[pivoted_df['State and County'].str.contains(' City$')])
print(merged_df.loc[merged_df['State and County'].str.contains('VA James')])

print(merged_df.loc[merged_df['State and County'].str.contains('Parish')])

for i in range(len(' Parish'),1,-1):
    pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace(f"{' Parish'[:i]}$", '', regex=True)

for i in range(len(' Municipality'),1,-1):
    pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace(f"{' Municipality'[:i]}$", '', regex=True)

pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Skagway'), 'COUNTYNAME'] = 'Skagway Municipality'


pivoted_df['State and County'] = pivoted_df['STATE'] + ' ' + pivoted_df['COUNTYNAME']


## Another check for counties not in merged/pivoted

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



## Setting St. to St in pivoted_df
pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('St. '), 'COUNTYNAME'] = pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('St. '), 'COUNTYNAME'].str.replace('St.', 'St')

# Replace "East" with "E" at the beginning of COUNTYNAME
pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace('^East ', 'E ', regex=True)

# Replace "West" with "W" at the beginning of COUNTYNAME
pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace('^West ', 'W ', regex=True)

# Display the updated DataFrame
print(pivoted_df.head())

city_of_counties = ['City of Buena Vista', 'City of Colonial Heights', 'City of Fairfax', 'City of Falls Church', 'City of Lexington', 'City of Manassas Park','City of Manassas', 'City of Petersburg', 'City of Staunton', 'City of Waynesboro','VA City of Fredericksburg','VA City of Roanoke','VA City of Franklin', 'VA City of Alexandria', 'VA City of Chesapeake', 'VA City of Hampton', 'VA City of Newport News', 'VA City of Norfolk', 'VA City of Portsmouth', 'VA City of Richmond', 'VA City of Roanoke', 'VA City of Suffolk', 'MD City of Baltimore', 'MO City of St. Louis', 'VA City of Virginia Beach']
for c in city_of_counties:
    # Find rows that match the condition
    matching_rows = pivoted_df.loc[pivoted_df['State and County'].str.contains(c[:2] + c[10:]) & pivoted_df['COUNTYNAME'].str.contains('city'), 'COUNTYNAME']
    
    # Check if there are any matching rows
    if not matching_rows.empty:
        pcounty = matching_rows.values[0]
        pcounty2 = pcounty.replace(' city', '')
        pcounty2 = "City of " + pcounty2
        pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains(pcounty) & pivoted_df['COUNTYNAME'].str.contains('city'), 'COUNTYNAME'] = pcounty2


for i in range(len(' city'),1,-1):
    pivoted_df['COUNTYNAME'] = pivoted_df['COUNTYNAME'].str.replace(f"{' city'[:i]}$", '', regex=True)


pivoted_df['State and County'] = pivoted_df['STATE'] + ' ' + pivoted_df['COUNTYNAME']


## Another check for counties not in merged/pivoted

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

## Dealing with Chugach and Copper River by combining them into Valdez-Cordova
print(merged_df.loc[merged_df['State and County'].str.contains('Valdez-Cordova')])
print(merged_df.columns)
print(merged_df.loc[merged_df['State and County'].str.contains('Valdez-Cordova'),'FIPS'])
merged_df[['FIPS','State and County']].head()

for column in pivoted_df.columns:
    print(column)

new_row = {
    'State and County': 'AK Valdez-Cordova',
    'STATEFIPS': 2,
    'STATE': 'AK',
    'COUNTYFIPS': 261,
    'COUNTYNAME': 'Valdez-Cordova',
}
colnames = []
# Add zero values for the rest of the columns
for column in pivoted_df.columns:
    if column not in new_row:
        new_row[column] = 0
        colnames.append(column)



# Convert the new row to a DataFrame
new_row_df = pd.DataFrame([new_row])

# Append the new row to the DataFrame using pd.concat
pivoted_df = pd.concat([pivoted_df, new_row_df], ignore_index=True)


# Display the updated DataFrame
print(pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Valdez-Cordova')])

pivoted_df['State and County'] = pivoted_df['STATE'] + ' ' + pivoted_df['COUNTYNAME']

vald_counties = ['Chugach','Copper River']
for c in vald_counties:
    for col in colnames:
        pivoted_df.loc[pivoted_df['State and County'].str.contains('Valdez-Cordova'), col] = int(pivoted_df.loc[pivoted_df['State and County'].str.contains('Valdez-Cordova'), col]) + int(pivoted_df.loc[pivoted_df['State and County'].str.contains(c), col])

pivoted_df = pivoted_df[~pivoted_df['COUNTYNAME'].isin(vald_counties)]


pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Prince of Wales-Hyd'), 'COUNTYNAME'] = 'Prince of Wales-Hyder'



pivoted_df['State and County'] = pivoted_df['STATE'] + ' ' + pivoted_df['COUNTYNAME']


## Another check for counties not in merged/pivoted

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






######## WORKING ON THE COUNTIES THAT ARE COMBINED IN MERGED_DF



############
## Doing Maui + Kalawao first
print(pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Maui')])
print(pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Kalawao')])
print(pivoted_df.loc[pivoted_df['STATE'].str.contains('HI')])
# Kalawao not included, so adding it to Maui for simplification (also probably that it was aggregated in tax measurment)
pivoted_df.loc[pivoted_df['COUNTYNAME'].str.contains('Maui'), 'COUNTYNAME'] = 'Maui + Kalawao'

pivoted_df.rename(columns={'COUNTYNAME': 'County'}, inplace=True)
pivoted_df.rename(columns={'STATE': 'State'}, inplace=True)

##########
## Also doing Roanoke + Salem because of Roanoke City

print(merged_df.loc[merged_df['State and County'].str.contains('Roanoke')])
print(merged_df.loc[merged_df['State and County'].str.contains('Salem')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Roanoke')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Salem')])


roanoke_values = pivoted_df.loc[pivoted_df['County'].str.contains('Roanoke') & ~(pivoted_df['State'].str.contains('City of'))].copy()
salem_values = pivoted_df.loc[pivoted_df['County'].str.contains('Salem') & pivoted_df['State'].str.contains('VA')].copy()

# Combine the values from Salem into Roanoke
for col in colnames:
    roanoke_values[col] = roanoke_values[col].sum() + salem_values[col].sum()

# Update the Roanoke row with the combined values
pivoted_df.loc[pivoted_df['County'].str.contains('Roanoke')& ~(pivoted_df['County'].str.contains('City of')), colnames] = roanoke_values[colnames]

# Remove the Salem row from merged_df
pivoted_df = pivoted_df[~((pivoted_df['County'] == ('Salem')) & pivoted_df['State'].str.contains('VA'))]

# Rename the Roanoke row to reflect the combined counties
pivoted_df.loc[pivoted_df['County'].str.contains('Roanoke')& ~(pivoted_df['County'].str.contains('City of')), 'County'] = 'Roanoke + Salem'

# Check the final merged_df
print(pivoted_df.loc[pivoted_df['County'].str.contains('Roanoke')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Salem')])

##########
## Also doing Southampton + Franklin VA because of confusion with Franklin City

print(pivoted_df.loc[pivoted_df['County'].str.contains('Franklin')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Southampton')])

southampton_values = pivoted_df.loc[pivoted_df['County'].str.contains('Southampton') & pivoted_df['State'].str.contains('VA')].copy()
franklin_values = pivoted_df.loc[pivoted_df['County'].str.contains('City of Franklin') & pivoted_df['State'].str.contains('VA')].copy()

# Combine the values from Southampton into Franklin
for col in colnames:
    franklin_values[col] = franklin_values[col].sum() + southampton_values[col].sum()

# Update the Franklin row with the combined values
pivoted_df.loc[pivoted_df['County'].str.contains('City of Franklin'), colnames] = franklin_values[colnames]

# Remove the Southampton row from pivoted_df
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Southampton')]

# Rename the Franklin row to reflect the combined counties
pivoted_df.loc[pivoted_df['County'].str.contains('City of Franklin'), 'County'] = 'Southampton + Franklin'

# Check the final pivoted_df
print(pivoted_df.loc[pivoted_df['County'].str.contains('Southampton')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Franklin')])

############
## Additionally doing Frederick + Winchester VA because of confusion with Fredericksburg

print(merged_df.loc[merged_df['State and County'].str.contains('Winchester')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Winchester')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Frederick')])

win_vals = pivoted_df.loc[pivoted_df['County'].str.contains('Winchester')].copy()
fred_vals = pivoted_df.loc[pivoted_df['County'].str.contains('Frederick')& pivoted_df['State'].str.contains('VA') & ~(pivoted_df['County'].str.contains('City of'))].copy()

# Combine the values from Winchester into Frederick
for col in colnames:
    fred_vals[col] = win_vals[col].sum() + fred_vals[col].sum()

# Update the Frederick row with the combined values
pivoted_df.loc[pivoted_df['County'].str.contains('Frederick')& pivoted_df['State'].str.contains('VA')& ~(pivoted_df['County'].str.contains('City of')), colnames] = fred_vals[colnames]

# Remove the Winchester row from pivoted_df
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Winchester')]

# Rename the Frederick row to reflect the combined counties
pivoted_df.loc[pivoted_df['County'].str.contains('Frederick')& pivoted_df['State'].str.contains('VA')& ~(pivoted_df['County'].str.contains('City of')), 'County'] = 'Frederick + Winchester'

print(pivoted_df.loc[pivoted_df['County'].str.contains('Frederick')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Winchester')])

############
## Additionally doing Spotsylvania + Fredericksburg VA because of confusion with Frederick
print(pivoted_df.loc[pivoted_df['County'].str.contains('Spotsylvania')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Fredericksburg')])

spot_vals = pivoted_df.loc[pivoted_df['County'].str.contains('Spotsylvania')].copy()
fred_vals = pivoted_df.loc[pivoted_df['County'].str.contains('Fredericksburg')].copy()

# Combine the values from Fredericksburg into Spotsylvania
for col in colnames:
    spot_vals[col] = spot_vals[col].sum() + fred_vals[col].sum()

# Update the Spotsylvania row with the combined values
pivoted_df.loc[pivoted_df['County'].str.contains('Spotsylvania'), colnames] = spot_vals[colnames]

# Remove the Fredericksburg row from pivoted_df
pivoted_df = pivoted_df[~pivoted_df['County'].str.contains('Fredericksburg')]

# Rename the Spotsylvania row to reflect the combined counties
pivoted_df.loc[pivoted_df['County'].str.contains('Spotsylvania'), 'County'] = 'Spotsylvania + Fredericksburg'

print(pivoted_df.loc[pivoted_df['County'].str.contains('Spotsylvania')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Fredericksburg')])


#########################
#########################
#########################
###########################
############################



pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


## Another check for counties not in merged/pivoted

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


combined_counties = [county for county in counties_not_in_gdp if '+' in county]

# Display the combined_counties list
print(combined_counties)
print(len(combined_counties))

######## Now doing the rest, which are all in VA
print(combined_counties)
combined_counties.sort_values(by='State',inplace=True)

for county in combined_counties:
    countyonly = county[3:]
    county1, county2 = countyonly.split(' + ')
    print(f"Combining {county1} + {county2}")
    county1_values = pivoted_df.loc[pivoted_df['County'].str.contains(county1) & pivoted_df['State'].str.contains('VA')].copy()
    county2_values = pivoted_df.loc[pivoted_df['County'].str.contains(county2) & pivoted_df['State'].str.contains('VA')].copy()

    for col in colnames:
        if col in county1_values.columns and col in county2_values.columns:
            county1_values[col] += county2_values[col].sum()

    pivoted_df.loc[pivoted_df['County'].str.contains(county1) & pivoted_df['State'].str.contains('VA')] = county1_values
    pivoted_df = pivoted_df[~(pivoted_df['County'].str.contains(county2) & pivoted_df['State'].str.contains('VA'))]
    pivoted_df.loc[pivoted_df['County'].str.contains(county1) & pivoted_df['State'].str.contains('VA'), 'County'] = f"{county1} + {county2}"
    print(pivoted_df.loc[pivoted_df['County'].str.contains(county2)])

pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']



###########################################################################################
## Final checks for combined counties in pivoted_df that arent in merged_df

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

###################
### More Specifics

# Dona Ana

pivoted_df.loc[pivoted_df['County'].str.contains('a Ana$'),'County'] = 'Dona Ana'














#############################################
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

original_counties = [
    'Greater Bridgeport', 
    'Southeastern Connect', 
    'South Central Connec', 
    'Lower Connecticut Ri', 
    'Northeastern Connect', 
    'Capitol Planning Reg', 
    'Naugatuck Valley Pla', 
    'Northwest Hills Plan', 
    'Western Connecticut'
]

equivalent_counties = [
    'Fairfield',        # Greater Bridgeport aligns closely with Fairfield County
    'New London',       # Southeastern Connecticut overlaps significantly with New London County
    'New Haven',        # South Central Connecticut corresponds to New Haven County
    'Middlesex',        # Lower Connecticut River has a relationship with Middlesex County
    'Windham',          # Northeastern Connecticut generally corresponds with Windham County
    'Hartford',         # Capitol Planning Region likely overlaps with Hartford County (Hartford is the Capitol)
    'New Haven',        # Naugatuck Valley Pla situated within New Haven County boundaries
    'Litchfield',       # Northwest Hills Plan encompasses towns part of Litchfield County
    'Fairfield'         # Western Connecticut largely corresponds to parts of Fairfield County
]

# Iterate over each county in equivalent_counties
for county in equivalent_counties:
    # Define the new row with the specified values
    new_row = {
        'County': county,
        'State': 'CT',
        'State and County': f'CT {county}'
    }
    
    # Add zero values for the rest of the columns
    for column in pivoted_df.columns:
        if column not in new_row:
            new_row[column] = 0
    
    # Convert the new row to a DataFrame
    new_row_df = pd.DataFrame([new_row])
    
    # Append the new row to the DataFrame using pd.concat
    pivoted_df = pd.concat([pivoted_df, new_row_df], ignore_index=True)

# Display the updated DataFrame to verify the changes
print(pivoted_df.tail())

for index in range(len(original_counties)):
    og = original_counties[index]
    eq = equivalent_counties[index]
    state = 'CT'
    og_val = pivoted_df.loc[pivoted_df['County'].str.contains(og) & pivoted_df['State'].str.contains(state)].copy()
    eq_val = pivoted_df.loc[pivoted_df['County'].str.contains(eq) & pivoted_df['State'].str.contains(state)].copy() 
    # Ensure that the columns to be combined are numeric
    colnames = [col for col in pivoted_df.columns if col not in ['State and County', 'County', 'State', 'COUNTYFIPS','STATEFIPS']]
        
    # Combine the values from eq_val into og_val
    for col in colnames:
        eq_val[col] = og_val[col].sum() + eq_val[col].sum()
        
    # Update the og_val row with the combined values
    pivoted_df.loc[pivoted_df['County'].str.contains(eq) & pivoted_df['State'].str.contains(state), colnames] = eq_val[colnames]
        
    # Remove the eq_val row from pivoted_df
    pivoted_df = pivoted_df[~(pivoted_df['County'].str.contains(og) & pivoted_df['State'].str.contains(state))]

print(pivoted_df.loc[pivoted_df['State'].str.contains('CT')])


#############################################
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

print(pivoted_df.loc[pivoted_df['County'].str.contains('Buena Vista')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Rockbridge')])


















##################### THESE CITIES MIGHT NOT BE UNDER "CITY OF X" SO CHECK EARLIER FOR THAT
original_counties = ['City of Buena Vista', 'City of Colonial Heights', 'City of Fairfax', 'City of Falls Church', 'City of Lexington', 'City of Manassas Park','City of Manassas', 'City of Petersburg', 'City of Staunton', 'City of Waynesboro']
equivalent_counties = ['Rockbridge', 'Dinwiddie', 'Fairfax', 'Arlington', 'Rockbridge', 'Prince William','Prince William', 'Dinwiddie', 'Augusta', 'Augusta']

for county in original_counties:
    print(pivoted_df.loc[pivoted_df['County'].str.contains(county[9:])])
for index in range(len(original_counties)):
    og = original_counties[index]
    eq = equivalent_counties[index]
    state = 'VA'
    og_val = pivoted_df.loc[pivoted_df['County'].str.contains(og) & pivoted_df['State'].str.contains(state)].copy()
    eq_val = pivoted_df.loc[pivoted_df['County'].str.contains(eq) & pivoted_df['State'].str.contains(state)].copy() 
    # Ensure that the columns to be combined are numeric
    colnames = [col for col in pivoted_df.columns if col not in ['State and County', 'County', 'State', 'COUNTYFIPS','STATEFIPS']]
        
    # Combine the values from eq_val into og_val
    for col in colnames:
        eq_val[col] = og_val[col].sum() + eq_val[col].sum()
        
    # Update the og_val row with the combined values
    pivoted_df.loc[pivoted_df['County'].str.contains(eq) & pivoted_df['State'].str.contains(state), colnames] = eq_val[colnames]
        
    # Remove the eq_val row from pivoted_df
    pivoted_df = pivoted_df[~(pivoted_df['County'].str.contains(og) & pivoted_df['State'].str.contains(state))]




print(pivoted_df.loc[pivoted_df['County'].str.contains('Rockbridge')])
print(pivoted_df.loc[pivoted_df['County'].str.contains('Buena Vista')])



pivoted_df['State and County'] = pivoted_df['State'] + ' ' + pivoted_df['County']


## Another check for counties not in merged/pivoted

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


#########
## No more counties in merged_df that arent in gdp_df
## Need to do last edits for counties:
## - CT Hartford
## - CT Litchfield
## - CT Tolland
## - IN DeKalb
## - LA LaSalle
## - MT Garfield
## - VA City of Roanoke
## - VA Franklin

# Hartford/CT Counties:
## might need to remove Hartford, Litchfield, Tolland from pivoted_df
## - Hartford, Litchfield, and Tolland are all major counties in CT; not likely to be merged with another county. Thus, can only 
##   assume they were left out for some reason. Thus, have to omit in pivoted

pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('CT') & pivoted_df['County'].isin(['Tolland', 'Hartford', 'Litchfield']))]

### Finishing the rest of those counties:

# LaSalle
# LaSalle not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('LA') & pivoted_df['County'].str.contains('LaSalle'))]

## DeKalb
# Dekalb not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('IN') & pivoted_df['County'].str.contains('DeKalb'))]

## Garfield
# Garfield not likely to be merged with another county, probably omitted; thus have to omit in pivoted
pivoted_df = pivoted_df[~(pivoted_df['State'].str.contains('MT') & pivoted_df['County'].str.contains('Garfield'))]


print(merged_df[merged_df.isna().any(axis=1)])
print(pivoted_df[pivoted_df.isna().any(axis=1)])


