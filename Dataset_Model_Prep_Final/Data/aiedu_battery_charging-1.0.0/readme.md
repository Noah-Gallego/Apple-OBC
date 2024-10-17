
# Format

A gzip of a large csv file containing a sample of battery charging data collected over time for different devices. The dataset contains two streams of information: changes in charging state (plugged in or unplugged), and battery charge level over time. The csv file contains 5 columns:
* `start` and `end`: timestamps (formatted as strings) that record the start and end of a plug event or battery charge level event
* `stream`: a string that indicates whether this data corresponds to a change in charging state (/device/isPluggedIn) or a change in battery charge level (/device/batteryPercentage)
* `value`: a value for plug state (0, unplugged, 1, plugged in) or battery charge level 0-100 percent
* `user_id`: anonymized, unique id's associated with each device in this dataset

This data can be used to learn about users' battery charging behaviors; information that is useful for optimizing battery charging over our devices. 

