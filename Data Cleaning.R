### Library Import ###
library(ggplot2)
library(dplyr)
library(DataExplorer)
library(VIM)
library(missForest)
library(tidyr)


##################################################################################################################
### Data Import ###
data <- read.csv("bank_dataset.csv")


##################################################################################################################
### Data Cleaning ###
# Replace all blank to NA
for (name in names(data)) {
  data[[name]] <- replace(data[[name]], data[[name]] == "", NA)
}

## Exploratory Data Analysis ##
# Plot missing value
plot_missing(data)

# checking_status (Missing value)
unique(data$checking_status)

# duration (Missing value, data inconsistent, outliers)
unique(data$duration)
ggplot(data, aes (duration)) +
  geom_boxplot()

# credit_history (Missing value)
unique(data$credit_history)

# purpose (Missing value)
unique(data$purpose)

# credit_amount (Missing value, data inconsistent, outliers)
unique(data$credit_amount)
ggplot(data, aes(credit_amount)) +
  geom_boxplot()

# saving_status (Missing value)
unique(data$savings_status)

# employement (Missing value)
unique(data$employment)

# installment_commitment (Missing value, data inconsistent)
unique(data$installment_commitment)
ggplot(data, aes(installment_commitment)) +
  geom_boxplot()

# personal_status (Missing value)
unique(data$personal_status)

# other_parties (Missing value)
unique(data$other_parties)

# residence_since (Missing value, data inconsistent)
unique(data$residence_since)
ggplot(data, aes(residence_since)) +
  geom_boxplot()

# property_magnitude (Missing value)
unique(data$property_magnitude)

# age (Missing value, data inconsitent, outliers)
unique(data$age)
ggplot(data, aes(age)) +
  geom_boxplot()

# other_payment_plans (Missing value)
unique(data$other_payment_plans)

# housing (Missing value)
unique(data$housing)

# existing_credit (Missing value, data inconsistent, outliers)
unique(data$existing_credits)
ggplot(data, aes(existing_credits)) +
  geom_boxplot()

# job (Missing value)
unique(data$job)

# num_dependents (Missing value, data inconsistent)
unique(data$num_dependents)
ggplot(data, aes(num_dependents)) +
  geom_boxplot()

# own_telephone (Missing value)
unique(data$own_telephone)

# foreign_worker (Missing value)
unique(data$foreign_worker)


##################################################################################################################
### Data Cleaning ###
## Random Forest Imputation ##
# Set seed
set.seed(1234)

# Change character to factor
for (name in names(data)) {
  if (is.character(data[[name]])) {
    data[[name]] <- as.factor(data[[name]])
  }
} 

# Random forest imputation
rf_imputation <- missForest(data)
imputed_data <- rf_imputation$ximp


##################################################################################################################
### Data Cleaning ###
## duration ##
# Round up to nearest duration
imputed_data$duration <- ceiling(imputed_data$duration)

# Show result
unique(imputed_data$duration)


## credit_amount ##
# Round off to 2 decimal places
imputed_data$credit_amount <- round(imputed_data$credit_amount, digits = 2)

# Show result
unique(imputed_data$credit_amount)


## saving_status ##
# Change 500<=X<10000 to 500<=X<1000
levels(imputed_data$savings_status)[levels(imputed_data$savings_status) == "500<=X<10000"] <- "500<=X<1000"

# Show result
unique(imputed_data$savings_status)

## installment_commitment ##
# Round off to 2 decimal places
imputed_data$installment_commitment <- round(imputed_data$installment_commitment, digits = 2)

# Show result
unique(imputed_data$installment_commitment)


## residence_since ##
# Round down to nearest integer
imputed_data$residence_since <- floor(imputed_data$residence_since)

# Show result
unique(imputed_data$residence_since)


## age ##
# Round down to nearest integer
imputed_data$age <- floor(imputed_data$age)

# Show result
unique(imputed_data$age)


## existing_credit ##
# Round off to whole number
imputed_data$existing_credits <- round(imputed_data$existing_credits, digits = 0)

# Show result
unique(imputed_data$existing_credits)


## num_dependents ##
# Round down to nearest integer
imputed_data$num_dependents <- floor(imputed_data$num_dependents)

# Show result
unique(imputed_data$num_dependents)


## own_telephone ##
# One hot encoding
imputed_data$own_telephone <- factor(imputed_data$own_telephone, levels = c("yes", "none"), labels = c("1", "0"))

# Show result
unique(imputed_data$own_telephone)


## foreign_worker ##
# One hot encoding
imputed_data$foreign_worker <- factor(imputed_data$foreign_worker, levels = c("yes", "no"), labels = c("1", "0"))

# Show result
unique(imputed_data$foreign_worker)


##################################################################################################################
### Data Export ###
write.csv(imputed_data, "cleaned_bank_set.csv")
