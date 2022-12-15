import pandas as pd
from geopy.geocoders import GoogleV3
import geopy.distance
import googlemaps

API='AIzaSyCE60a-gwat-zDw3eEOfmagrFg6RqDZa40'
gmap = googlemaps.Client(key=API)
#print(type(gmap))
data = pd.read_excel(r'/Users/seraph2019/Desktop/GeocodingAnalysis/Full Address.xlsx',"Sheet1")
#print(data)

#Find out travel distance between addresses
origins = data["Address"] #Let's say this is the origin
#print(origins)
#print(type(origins))
#destinationCoordinate = (42.3917712,-83.50460439999999)

#Calculating distance
actual_distance = []
for origin in origins:
    result= gmap.distance_matrix("14200 N Haggerty Rd, Plymouth, MI 48170", origin, mode = 'driving')["rows"][0]["elements"][0]["distance"]["value"]
    print(result)
    result = result/1609.43 # Transfer to Miles
    actual_distance.append(result)

#Calculating duration
#Assuming arrival time is Sep 20th 2021
actual_duration = []
for origin in origins:
    result = gmap.distance_matrix("14200 N Haggerty Rd, Plymouth, MI 48170", origin, mode = 'driving', arrival_time = 1632722400 )["rows"][0]["elements"][0]["duration"]["value"]
    print(result)
    minutes = result/60
    actual_duration.append(minutes)

#Add the list of coordinates to the main data set
data["duration (Minutes)"] = actual_duration
#data.head(10)
#Add the list of coordinates to the main data set
data["distance (Miles)"] = actual_distance
#data.head(15)

writer = pd.ExcelWriter("/Users/seraph2019/Desktop/DT1time.xlsx", engine = 'xlsxwriter')
# Write your DataFrame to a file
# yourData is a dataframe that you are interested in writing as an excel file
data.to_excel(writer, 'Sheet2')
print(data)
# Save the result
writer.save()
