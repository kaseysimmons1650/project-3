import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('troop_movements.csv')

#print(data.head())

empire_or_resistance = data.groupby("empire_or_resistance").count()
columns_to_remove = ['timestamp', 'unit_id', 'unit_type', 'location_x', 'location_y','destination_x', 'destination_y']
df_filtered = empire_or_resistance.drop(columns=columns_to_remove)
empire_res_count = df_filtered.rename(columns={'homeworld': 'count', 'empire_or_resistance': 'test'})

#final empire / res count
print(empire_res_count)


plt.figure(figsize=(10, 6))
sns.barplot(x="empire_or_resistance", y="count", data=empire_res_count)
plt.title('Character Count by Empire or Resistance')
# plt.xlabel('Empire or Resistance')
# plt.ylabel('Count')
plt.show()



# char_by_homeworld = data.groupby("homeworld").count()
# # print(char_by_homeworld)
# columns_to_remove = ['timestamp', 'unit_id', 'unit_type', 'location_x', 'location_y','destination_x', 'destination_y']
# df_filtered = char_by_homeworld.drop(columns=columns_to_remove)
# char_homeworld = df_filtered.rename(columns={'empire_or_resistance': 'count'})


# #filtered homeworld count by character
# print(char_homeworld)



# char_unit_type = data.groupby("unit_type").count()
# columns_to_remove = ['timestamp', 'unit_id', 'location_x', 'location_y','destination_x', 'destination_y', 'homeworld']
# df_filtered = char_unit_type.drop(columns=columns_to_remove)
# unit_type_count= df_filtered.rename(columns={'empire_or_resistance': 'count'})

# #filtered unit type count
# print(unit_type_count)


#print(len(data))
data["is_resistance"] = data["empire_or_resistance"].apply(lambda x: True if x == "resistance" else False)

# columns_to_remove = ['timestamp', 'unit_id', 'location_x', 'location_y','destination_x', 'destination_y', 'homeworld']
# df_filtered = data.drop(columns=columns_to_remove)

# print(df_filtered)




# #select feature and target values
# features = ['homeworld', 'unit_type']
# target = 'is_resistance'
# x = data[features]
# y = data[target]

# #conver cat feature to numeric using one hot encoding
# x_encoded = pd.get_dummies(x)

# # importances = model.feature_importances_

# # Handle missing values
# # df = df.dropna(subset=['Age'])  # Drop rows with missing 'Age' or 'Embarked'
# # Using pd.get_dummies()
# # df_dummies = pd.get_dummies(data, columns=['Sex'], drop_first=True)


# # Using OneHotEncoder as an alternative
# encoder = OneHotEncoder(drop='first', sparse_output=False)
# encoded_features = encoder.fit_transform(df[['Sex']])
# encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Sex']))

# # Concatenate encoded features with the original dataframe
# df_encoded = pd.concat([df.drop(columns=['Sex']), encoded_df], axis=1)

# # Define features (X) and target (y)
# print(x_encoded.head())
# print(x_encoded.columns)
# X = x_encoded[['Pclass', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare', 'Sex_male']]
# y = x_encoded['Survived']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize the Decision Tree Classifier
# clf = DecisionTreeClassifier()

# # Train the classifier
# clf.fit(X_train, y_train)

# # Make predictions on the testing set
# y_pred = clf.predict(X_test)

# # Evaluate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # Plot the decision tree
# plt.figure(figsize=(20,10))
# tree.plot_tree(clf, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
# plt.title("Decision Tree Visualization")
# plt.show()







# Select the features and target variable
features = ['homeworld', 'unit_type']
target = 'is_resistance'
X = data[features]  # Features
y = data[target]  # Target variable

# Convert categorical features to numeric using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to hold the feature importances
feature_importances = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})

# Sort the DataFrame by importance in descending order
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

# Display the most influential unit_type
most_influential_unit_type = feature_importances.iloc[0]['Feature']
print('Most Influential Unit Type:', most_influential_unit_type)




# Save trained model
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

