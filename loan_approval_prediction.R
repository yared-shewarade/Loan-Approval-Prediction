# 1. Install and load necessary packages
install.packages(c("tidyverse", "dplyr", "tidyr"))
install.packages("caret")
install.packages("tm")
install.packages("languageserver")

# Loading libraries
library(caret)
library(tidyverse)
library(tm)
library(dplyr)
library(tidyr)

# 2. load dataset into a data frame
loan_data <- read.csv("dataset.csv")

# 3. Explore the data
# View the structure of the data
str(loan_data)

# Display summary statistics
summary(loan_data)
loan_data