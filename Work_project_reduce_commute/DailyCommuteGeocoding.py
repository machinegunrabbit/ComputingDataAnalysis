import pandas as pd
from geopy.geocoders import GoogleV3
import geopy.distance
import googlemaps

API='AIzaSyCE60a-gwat-zDw3eEOfmagrFg6RqDZa40'#Transforming addresses to long/latitude (write a for loop)
geolocator = GoogleV3(api_key=API)#geolocator
#print(type(geolocator))
data = pd.read_excel(r'/Users/seraph2019/Desktop/Full Address.xlsx')#Loading location data from excel to py
address_lst = data["Address"].tolist()
#print (address_lst)

#Imputing addresses information
long_list = []
la_list = []
merged_lst = []
#c = 0
for address in address_lst:
    location = geolocator.geocode(address)
    print(location)
    long_list.append(location.longitude)
    la_list.append(location.latitude)
    #c = c + 1

# Creating Coordinates column
merged_lst = list(zip(la_list, long_list))
print(long_list)
print(la_list)
print(merged_lst)

data["Longitude"] = long_list
data["Latitude"] = la_list
data["Coordinates"] = merged_lst

# Specify a writer
writer = pd.ExcelWriter("/Users/seraph2019/Desktop/lgla.xlsx", engine = 'xlsxwriter')
# Write your DataFrame to a file
# yourData is a dataframe that you are interested in writing as an excel file
data.to_excel(writer, 'Sheet2')
print(data)
# Save the result
writer.save()
