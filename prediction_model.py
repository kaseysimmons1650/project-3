import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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





