library(DoubleML)
library(hdm)
library(glmnet)
library(xgboost)
library(data.table)
library(Matrix)
library(ggplot2)
library(tidyverse)
library(dplyr)


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

# Automate subset creation
for (i in seq_len(nrow(linecode_mapping))) {
  assign(
    linecode_mapping$var_name[i],
    cainc30 %>% filter(LineCode == linecode_mapping$LineCode[i])
  )
}

print(cagdp8 %>% filter(GeoFIPS == "\\d\\d000"))
