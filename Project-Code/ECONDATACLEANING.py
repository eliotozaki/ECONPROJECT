import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap
from pandas_profiling import ProfileReport

population_df = pd.read_csv("C:/Users\eliot/OneDrive/Desktop/ECONPROJECT/Cleaned-Datasets/POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Base-Datasets/COUNTY_EMISSIONS2021.csv")


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

states = list(state_initials.keys())
initials = list(state_initials.values())
population_df.tail(20)

for state in states:
    population_df.loc[population_df['State']==state,'State'] = state_initials[state]
    
print(population_df['State'].unique())
print(population_df['County'][population_df['County'].apply(len) <= 2])

index = 0
popcounts = 0
for initial in initials:
    count = (population_df['State'] == initial).sum()
    print(f"{initial}: {count}")

for initial in initials:
    count = (emissions_df['State'] == initial).sum()
    print(f"{initial}: {count}")


population_df = pd.read_csv("C:/Users\eliot/OneDrive/Desktop/ECONPROJECT/Cleaned-Datasets/POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("C:/Users/eliot/OneDrive/Desktop/ECONPROJECT/Base-Datasets/COUNTY_EMISSIONS2021.csv")

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

states_with_differences = merged_df['State'][merged_df['Difference'] != 0].tolist()

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

emissions_df.set_index("County")
population_df.set_index("County")

df = pd.merge(population_df,emissions_df)

df.describe()
df.columns

population_df.to_csv("POPULATIONDATA-Cleaned.csv")
emissions_df.to_csv("EMISSIONSDATA-Cleaned.csv")
df.to_csv("AllData-MergedDS.csv")




#Cleaning More Datasets
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap
#from pandas_profiling import ProfileReport

population_df = pd.read_csv("Cleaned-Datasets\POPULATIONDATA-Cleaned.csv")
emissions_df = pd.read_csv("Cleaned-Datasets/EMISSIONSDATA-Cleaned.csv")
merged_df = pd.read_csv("AllData-MergedDS.csv")

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

states = list(state_initials.keys())
initials = list(state_initials.values())

import chardet
with open("Cleaned-Datasets\County-MSA-GDP-DATA.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

gdp_df = pd.read_csv("Cleaned-Datasets\County-MSA-GDP-DATA.csv",encoding="Windows-1252")
gdp_df.describe()

gdp_df.head(10)

gdp_df = gdp_df[~gdp_df['GeoName'].isin(states + ['United States'])]

print(len(gdp_df)/3)
counties_not_in_gdp = set(merged_df['County'].str.split(',').str[0]) - set(gdp_df['GeoName'].str.split(',').str[0])
print("Counties in merged_df but not in gdp_df:")
for county in counties_not_in_gdp:
    print(county)

counties_not_in_merged = set(gdp_df['GeoName'].str.split(',').str[0]) - set(merged_df['County'].str.split(',').str[0])
print("Counties in gdp but not in merged:")
for county in counties_not_in_merged:
    print(county)

gdp_df.loc[gdp_df['GeoName'].str.contains(r' \(Independent City\)', regex=True), 'GeoName'] = \
    gdp_df['GeoName'].str.replace(r' \(Independent City\)', '', regex=True).str.strip().apply(lambda x: "City of " + x)
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('St.', 'St')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('East', 'E')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('Census Area', '')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('City and Borough', '')
gdp_df['GeoName'] = gdp_df['GeoName'].str.replace('Borough', '')
gdp_df.loc[gdp_df['GeoName'] =='LaSalle', 'GeoName'] = 'La Salle'

print(gdp_df.loc[gdp_df['GeoName'].str.contains('Emporia'),'GeoName'])
print(merged_df.loc[merged_df['County'].str.contains('Columbia'),'County'])
print(gdp_df.loc[~gdp_df['GeoName'].str.contains(','),'GeoName'])

gdp_df.set_index("GeoName")
print(merged_df.loc[merged_df['County'].str.contains('Rocky Mountain'),'County'])


pivoted_df = gdp_df.pivot_table(index=['GeoName', 'GeoFIPS'], columns='Unit', values='GDP(Thousands)', aggfunc='first')

# Flatten the column MultiIndex, if necessary
pivoted_df.columns = [col for col in pivoted_df.columns]
# Reset the index to turn GeoName back into a column
pivoted_df.reset_index(inplace=True)
print(pivoted_df)


# Separate GeoName into County and State columns
pivoted_df[['County', 'State']] = pivoted_df['GeoName'].str.split(', ', expand=True).iloc[:, [0, 1]]
pivoted_df['County'] = pivoted_df['County'].str.rstrip()
#geosplit = pivoted_df['GeoName'].str.split(', ', expand=True)
#geosplit.shape
#geosplit.head()
#geosplit[:-1].head()
pivoted_df.head(10)
pivoted_df.shape
pivoted_df.set_index("County")
pivoted_df.sort_values(["State", "County"], inplace=True)
merged_df.sort_values(["State", "County"], inplace=True)
merged_df.columns

merged_df[['County','State']].head(10)
pivoted_df[['County','State']].head(10)




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

states_with_differences = merged_counts_df['State'][merged_counts_df['Difference'] != 0].tolist()
print(states_with_differences)

na_rows = pivoted_df[pivoted_df['State'].isna()]
print(na_rows)
nostate_counties = na_rows['County'].unique()
print(merged_df[merged_df['County'].isin(nostate_counties)][['County', 'State']])
pivoted_df.loc[pivoted_df['County']=="District of Columbia", 'State'] = 'DC'
pivoted_df = pivoted_df.dropna(subset=['State'])
pivoted_df.loc[pivoted_df['County'].str.startswith('West '), 'County'] = pivoted_df['County'].str.replace('West', 'W')

#######CANNOT RUN YET, NEED TO ELIM NAS IN PIVOTED_DF
#counties_with_asterisk = pivoted_df[pivoted_df['State'].str.contains('\*')][['County', 'State']]
#print(counties_with_asterisk)

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

print(pivoted_df[pivoted_df['State'].str.contains('\*')])
edited_counties = pivoted_df[pivoted_df['State'].str.contains('\*')]

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
with open('/mnt/data/geo_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(["Code", "State", "County"])
    
    # Write the data rows
    for code, (state, county) in geo_dict.items():
        writer.writerow([code.strip(), state, county.strip()])