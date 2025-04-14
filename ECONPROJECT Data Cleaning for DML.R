library(DoubleML)
library(hdm)
library(glmnet)
library(xgboost)
library(data.table)
library(Matrix)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(stringr)


# Load the data
cagdp1 = read.csv("Panel_Data/GDP-income/CAGDP1__ALL_AREAS_2001_2022.csv")
cagdp2 = read.csv("Panel_Data/GDP-income/CAGDP2__ALL_AREAS_2001_2022.csv")
cagdp8 = read.csv("Panel_Data/GDP-income/CAGDP8__ALL_AREAS_2001_2022.csv")
cainc1 = read.csv("Panel_Data/GDP-income/CAINC1__ALL_AREAS_1969_2022.csv")
cainc30 = read.csv("Panel_Data/GDP-income/CAINC30__ALL_AREAS_1969_2022.csv")

# Chainging colnames of x2001 - x2022 to 2001 - 2022
cagdp1 <- cagdp1 %>% rename_with(~str_replace(., "X", ""), starts_with("X"))
cagdp2 <- cagdp2 %>% rename_with(~str_replace(., "X", ""), starts_with("X"))
cagdp8 <- cagdp8 %>% rename_with(~str_replace(., "X", ""), starts_with("X"))
cainc1 <- cainc1 %>% rename_with(~str_replace(., "X", ""), starts_with("X"))
cainc30 <- cainc30 %>% rename_with(~str_replace(., "X", ""), starts_with("X"))

summary(cagdp1)

## Removing spaces at front of GeoFIPS
dfs = list(cagdp1, cagdp2, cagdp8, cainc1, cainc30)

for (i in seq_along(dfs)) {
  df = dfs[[i]]
  df$GeoFIPS <- as.character(df$GeoFIPS)
  df$GeoFIPS <- df$GeoFIPS %>%
    gsub("\\\\", "", .) %>%           # Remove backslashes
    gsub('"', "", .) %>%             # Remove double quotes
    gsub("^\\s+|\\s+$", "", .)
  df_filtered = df %>% filter(grepl("^\\d{2}000$", GeoFIPS))
  
  filtered_geo_fips <- unique(df_filtered$GeoFIPS)
  print(filtered_geo_fips)
  
  #removing all observations with a GeoFIPS in filtered_geo_fips
  dfs[[i]] = df %>% filter(!GeoFIPS %in% filtered_geo_fips) 
}

cagdp1 = dfs[[1]]
cagdp2 = dfs[[2]]
cagdp8 = dfs[[3]]
cainc1 = dfs[[4]]
cainc30 = dfs[[5]]

summary(cagdp1)


### Finding what counties are in cainc30 and not in cagdp1
cainc30_counties = unique(cainc30$GeoFIPS)
cagdp1_counties = unique(cagdp1$GeoFIPS)

cainc30_not_in_cagdp1 = cainc30_counties[!cainc30_counties %in% cagdp1_counties]
print(cainc30_not_in_cagdp1)
print(cainc30[cainc30$GeoFIPS %in% cainc30_not_in_cagdp1, "GeoName"])
cagdp1 %>% filter("AK*" %in% GeoName)

# Differences found in AK and WI county groupings when surveying
cagdp1 %>% filter(grepl("^02\\d{3}",GeoFIPS)) %>% select("GeoName")


#### Making State and County Columns

# Extracting state and county from GeoFIPS
cagdp1$State <- str_extract(cagdp1$GeoName, "[A-Z]{2}")
cagdp1$County <- str_extract(cagdp1$GeoName, "^[^,]+")

cagdp2$State <- str_extract(cagdp2$GeoName, "[A-Z]{2}")
cagdp2$County <- str_extract(cagdp2$GeoName, "^[^,]+")

cagdp8$State <- str_extract(cagdp8$GeoName, "[A-Z]{2}")
cagdp8$County <- str_extract(cagdp8$GeoName, "^[^,]+")

cainc1$State <- str_extract(cainc1$GeoName, "[A-Z]{2}")
cainc1$County <- str_extract(cainc1$GeoName, "^[^,]+")

cainc30$State <- str_extract(cainc30$GeoName, "[A-Z]{2}")
cainc30$County <- str_extract(cainc30$GeoName, "^[^,]+")

## TO DO####################################################################################
# Housekeeping for counties with alternate characters in name (e.g. n with tildae)
#dfs = list(cagdp1, cagdp2, cagdp8, cainc1, cainc30)



#for (df in dfs) {
  # Changing all "<f1>" to n:
#  df$County <- gsub("\xf1", "n", df$County)
  
#}
#############################################################################################################


### Cleaning Alaska Data


print(cagdp1 %>% filter(State == "AK") %>% select("GeoName"))
print(cainc1 %>% filter(State == "AK") %>% select("GeoName"))

cagdp1_ak_counties = cagdp1 %>% filter(State == "AK") %>% select("County") %>% distinct()
cainc1_ak_counties = cainc1 %>% filter(State == "AK") %>% select("County") %>% distinct()

print(cagdp1_ak_counties)
print(cainc1_ak_counties)

print(setdiff(cagdp1_ak_counties$County, cainc1_ak_counties$County))
print(setdiff(cainc1_ak_counties$County, cagdp1_ak_counties$County))





## Removing all rows where there is a value "(NA)" in the data

# Display all rows with NA values in any column
rows_with_na <- cagdp1 %>% filter(if_any(everything(), is.na))

# Print the rows with NA
print(rows_with_na)
year_columns = colnames(cainc1)[9:(ncol(cainc1)-2)]

print(cainc1 %>% filter(if_all(all_of(year_columns),~ . == "(NA)")))
# REMOVING ALL OBS. WITH ALL NA VALUES
cainc1 = cainc1 %>% filter(!if_all(all_of(year_columns),~ . == "(NA)"))


print(cagdp1 %>% filter(State == "AK") %>% select("County") %>% distinct())
print(cainc1 %>% filter(State == "AK") %>% select("County") %>% distinct())


# Now looking at the same thing in cainc30
year_columns = colnames(cainc30)[9:(ncol(cainc30)-2)]
print(cainc30 %>% filter(if_all(all_of(year_columns),~ . == "(NA)")))




# Separating cagdp1 by linecode/description
cagdp1_realgdp = cagdp1 %>% filter(LineCode == 1)
cagdp1_quantindex = cagdp1 %>% filter(LineCode == 2)
cagdp1_currdollgdp = cagdp1 %>% filter(LineCode == 3)

summary(cagdp1_realgdp)

# Separating cagdp2 by linecode/description

# Using regex to remove extra digits at the end of descriptions
cagdp2$Description <- gsub("\\s\\d+/$", "", cagdp2$Description)

# Create a mapping of LineCode to Description
linecode_description_mapping <- cagdp2 %>%
  select(LineCode, Description) %>%  # Select relevant columns
  distinct() %>%  # Remove duplicates
  arrange(LineCode)  # Sort by LineCode for clarity

# Print the mapping
print(linecode_description_mapping)

# Create a mapping of LineCode to variable names
linecode_mapping <- linecode_description_mapping %>%
  filter(!is.na(LineCode)) %>%  # Exclude NA LineCode
  mutate(var_name = paste0("cagdp2_", gsub(",","",gsub(" ", "_", tolower(trimws(Description))))))  # Generate variable names

tail(linecode_mapping)
tail(cagdp2 %>% filter(is.na(LineCode)))
cagdp2 = cagdp2 %>% filter(!is.na(LineCode))
# Automate subset creation
for (i in seq_len(nrow(linecode_mapping))) {
  assign(
    linecode_mapping$var_name[i],
    cagdp2 %>% filter(LineCode == linecode_mapping$LineCode[i])
  )
}


# Separating cagdp8 by linecode/description

# Using regex to remove extra digits at the end of descriptions
cagdp8$Description <- gsub("\\s\\d+/$", "", cagdp8$Description)

# Create a mapping of LineCode to Description
linecode_description_mapping <- cagdp8 %>%
  select(LineCode, Description) %>%  # Select relevant columns
  distinct() %>%  # Remove duplicates
  arrange(LineCode)  # Sort by LineCode for clarity

# Print the mapping
print(linecode_description_mapping)

# Create a mapping of LineCode to variable names
linecode_mapping <- linecode_description_mapping %>%
  filter(!is.na(LineCode)) %>%  # Exclude NA LineCode
  mutate(var_name = paste0("cagdp8_", gsub(",","",gsub(" ", "_", tolower(trimws(Description))))))  # Generate variable names

head(linecode_mapping)
tail(cagdp8 %>% filter(is.na(LineCode)))

cagdp8 = cagdp8 %>% filter(!is.na(LineCode))

# Automate subset creation
for (i in seq_len(nrow(linecode_mapping))) {
  assign(
    linecode_mapping$var_name[i],
    cagdp8 %>% filter(LineCode == linecode_mapping$LineCode[i])
  )
}

# Separating cainc1 by linecode/description
cainc1_personal_income = cainc1 %>% filter(LineCode == 1)
cainc1_population = cainc1 %>% filter(LineCode == 2)
cainc1_personal_income_per_capita = cainc1 %>% filter(LineCode == 3)

# Separating cainc30 by linecode/description

# Using regex to remove extra digits at the end of descriptions
cainc30$Description <- gsub("\\s\\d+/$", "", cainc30$Description)

# Create a mapping of LineCode to Description
linecode_description_mapping <- cainc30 %>%
  select(LineCode, Description) %>%  # Select relevant columns
  distinct() %>%  # Remove duplicates
  arrange(LineCode)  # Sort by LineCode for clarity

# Print the mapping
print(linecode_description_mapping)

# Create a mapping of LineCode to variable names
linecode_mapping <- linecode_description_mapping %>%
  filter(!is.na(LineCode)) %>%  # Exclude NA LineCode
  mutate(var_name = paste0("cainc30_", gsub(",","",gsub(" ", "_", tolower(trimws(Description))))))  # Generate variable names

head(linecode_mapping)
tail(cainc30 %>% filter(is.na(LineCode)))

cainc30 = cainc30 %>% filter(!is.na(LineCode))


##################
## Cleaning CAINC30

# Finding differences:

### Compare Counties Between cagdp1_realgdp and cainc30_average_wages_and_salaries
# Find counties in cainc30_average_wages_and_salaries but not in cagdp1_realgdp
counties_in_cainc30_not_cagdp1 <- anti_join(
  cainc30, 
  cainc1_personal_income, 
  by = c("State", "County")
) %>%
  select(State, County, GeoFIPS, GeoName) %>%
  distinct()

# Find counties in cagdp1_realgdp but not in cainc30_average_wages_and_salaries
counties_in_cagdp1_not_cainc30 <- anti_join(
  cainc1_personal_income, 
  cainc30, 
  by = c("State", "County")
) %>%
  select(State, County, GeoFIPS, GeoName) %>%
  distinct()

# Print results
cat("\nCounties in cainc30_average_wages_and_salaries but not in cagdp1_realgdp:\n")
print(counties_in_cainc30_not_cagdp1)
cat("\nCounties in cagdp1_realgdp but not in cainc30_average_wages_and_salaries:\n")
print(counties_in_cagdp1_not_cainc30)

# Summarize counts
cat("\nNumber of counties in cainc30_average_wages_and_salaries but not in cagdp1_realgdp:", 
    nrow(counties_in_cainc30_not_cagdp1), "\n")
cat("Number of counties in cagdp1_realgdp but not in cainc30_average_wages_and_salaries:", 
    nrow(counties_in_cagdp1_not_cainc30), "\n")

cainc30 = cainc30 %>% filter(GeoFIPS %in% cainc1_personal_income$GeoFIPS)

#################



# Automate subset creation
for (i in seq_len(nrow(linecode_mapping))) {
  assign(
    linecode_mapping$var_name[i],
    cainc30 %>% filter(LineCode == linecode_mapping$LineCode[i])
  )
}

print(linecode_mapping)

## Printing dims
print(dim(cagdp1))

## Finding differences between CAINC30 data frames and other data frames
tail(cagdp1_realgdp)
tail(`cainc30_population_(persons)`)








########Finding differences between cagdp1_realgdp and cainc1_personal_income
# Find counties in cainc1_personal_income but not in cagdp1_realgdp

counties_in_cainc1_not_cagdp1 <- anti_join(
  cainc1_personal_income, 
  cagdp1_realgdp, 
  by = c("State", "County")
) %>%
  select(State, County, GeoFIPS, GeoName) %>%
  distinct()

# Find counties in cagdp1_realgdp but not in cainc1_personal_income
counties_in_cagdp1_not_cainc1 <- anti_join(
  cagdp1_realgdp, 
  cainc1_personal_income, 
  by = c("State", "County")
) %>%
  select(State, County, GeoFIPS, GeoName) %>%
  distinct()

# Print results
cat("\nCounties in cainc1_personal_income but not in cagdp1_realgdp:\n")
print(counties_in_cainc1_not_cagdp1)
cat("\nCounties in cagdp1_realgdp but not in cainc1_personal_income:\n")
print(counties_in_cagdp1_not_cainc1)
# Summarize counts
cat("\nNumber of counties in cainc1_personal_income but not in cagdp1_realgdp:", 
    nrow(counties_in_cainc1_not_cagdp1), "\n")
cat("Number of counties in cagdp1_realgdp but not in cainc1_personal_income:",
    nrow(counties_in_cagdp1_not_cainc1), "\n")

print(cagdp1_realgdp %>% filter(State == "AK")) %>% select("GeoName")






### TO DO: Clean rest of CAGDP1, CAGDP8, Etc to match CAINC1 (use CAINC1_personal_income)