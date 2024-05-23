from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
X.to_csv('features.csv')
y = heart_disease.data.targets
y.to_csv('targets.csv')

# metadata
metadata = heart_disease.metadata
print(metadata)

# variable information
variable_information = heart_disease.variables
variable_information.to_csv('variable_information.csv')
