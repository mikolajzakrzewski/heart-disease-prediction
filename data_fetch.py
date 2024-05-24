from ucimlrepo import fetch_ucirepo

# fetch dataset
dataset = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = dataset.data.features
X.to_csv('dataset/features.csv')
y = dataset.data.targets
y.to_csv('dataset/targets.csv')

# metadata
metadata = dataset.metadata
print(metadata)

# variable information
variable_information = dataset.variables
variable_information.to_csv('dataset/variable_information.csv')
