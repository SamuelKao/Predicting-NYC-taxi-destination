# Where are you going?

## Data Download

* **Data Source:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
* **Data Description:** On the website you will find “Yellow Taxi Trip Records (PARQUET)” download links for every month of 2023, download these and place them on your google drive in a “CSE151GP” folder in a “data” subfolder (My Drive/CSE151GP/data/ is the relative path). The notebook will then mount to your drive and place this data into pandas dataframes once you run the program.
* **Taxi Zone Data source** https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc
* **Taxi Zone Data Description** Export the data into a csv file and open it with google sheets and make sure it has the name "Location_ID_encoder.xlsx". Then place the xlsx file in the same data folder as with the parquet files of the data. The notebook will then be able to read this file as well once you have mounted your drive.

## Environment Setup
The environment being used is the default google colab environment from 11-03-2024, but with the “holidays” package installed with pip. For future reproducibility we have attached a requirements.txt file.
1. **Python Version:** 3.10.12
2. **Required Libraries:**
    * holidays==0.59
    * Default google colab libraries from 11-03-2024

3. **Installation:**
    ```bash
    pip install -r requirements.txt 
    ```

## Preprocessing

### Data Description
The data consists of a 12 parquet files with data from all cabs in New York in 2023. We then load these files into data frames, each represententing a different month of the year, and in total it is around 38 million data points. Every datapoint has 19 features of different kinds of formats ranging from datetime, int, float and object, meaning that some features need to be encoded in order to be useful. The data is retrieved from The Taxi & Limousine Commission (TLC) and includes some data points with missing features and outliers and therefore the data need to be cleaned before it can be used. 


### Relevant data
Our model is meant to be used before the cab starts riding, meaning that a lot of features are "unknown". The features we can use are then pickup time(tpep_pickup_datetime), passenger count and pickup location(PULocationID), while dropoff location(DOLocationID) is the feature we want to predict. 

tpep_pickup_datetime is formatted as a datetime64 and must be encoded to effectively incorporate date and time information into our model. First, we plan to apply **cyclical encoding** to both day and time. Specifically, we’ll convert the datetime columns into two features: one representing the number of days since the start of the year and another representing seconds since the start of the day. By then applying sine and cosine transformations to these values, we can capture the cyclical nature of time, allowing the model to understand that day 365 is close to day 1 and that 23:59:59 is close to 00:00:00. The resulting values will be scaled between -1 and 1 for all four columns. 
We also plan to encode the **day of the week** using ordinal encoding (e.g., Monday=0, Tuesday=1, etc.), which preserves the natural order of days while scaling from 0 to 6 as integer values. This will help the model interpret weekday patterns directly. In addition, we intend to include **holiday features**. Specifically, we’ll create indicators for whether a day is the day before a holiday, an actual holiday, or the day after a holiday. If a day meets any of these criteria, we will note the specific holiday. At this stage, our tentative plan is to encode these indicators as true/false values, as the time of year may already provide sufficient context regarding the holiday itself. This means that the resulting scale for holiday features will remain minimal.

The feature passenger_count might not be a strong predictor on its own but it could offer insights into the destination type. For example, taxis carrying more passengers may be heading towards popular locations that accommodate larger groups, such as tourist spots, hotels, airports, or event venues. Therefore we plan to feed it into the model. 
For PULocationID and DOLocationID the locations in the dataset are encoded with a location id 1 to 262 that can be mapped to a location in New York and this data is found separately in an encoding sheet. The location data consists of Shape_Leng, the_geom, Shape_Area, zone, LocationID, and borough. Zone is a string with the name of the location, useful to understand the data and borough is a string with the district of the location. LocationID is the location id that we will map together with our main data. Shape_Leng, the_geom, Shape_Area together describe a polygon of the location area in NY that is correlated to the specific location id. This data consists only of 262 entries with location 56 repeated twice and 103 trice, while locationID 104, 105, 57 is not included at all. Looking at other maps of the zone codes in NYC, it is apparent that this is just a mistake in the location ID data that can easily be fixed. 

Since the model is a classification model, we plan to encode the drop off location ids with one hot encoding. This will also help us later to get the 5 most probable drop off locations. For the pickup location however, we want to encode the location to make sure the model understands which locations are close together. To do this we will use ordinal encoding based on the coordinates from the polygon description in the encoding sheet. For example we could sort them north to south or maybe both north to south and west to east to give the model more features to work with. 

### Missing data
Missing data have been found for the features passenger_count, RatecodeID, store_and_fwd_flag, congestion_surcharge and Airport_fee. Since the only feature we will use out of these is passenger_count we only need to handle this one. We plan to use median imputation which for this data is 1 passenger, this will be both for when the data is null and 0 since 0 passengers would not count as a cab ride. 

### Outliers
While exploring the data, it was noticed that there are some dates that there were about 100 rides not in the year of 2023, these data points will be dropped. There are also 2475 sets of passengers who were dropped off before they were picked up, and these data points will also be dropped. The same goes for the 30051 rides longer than 10 hours, since we believe that these stem from taxi drivers forgetting to turn off the taximeter, and 10 hours is the maximum amount of time a taxi driver is allowed to work within a 24 hour window. We also consider trips with a trip distance that is unrealistic short to be outliers and remove those. 
Since the data we will use needs to be encoded, it is not possible to create any correlation or scatterplot of the relevant data at this point. There is a pairplot in the code, but this will need to be addressed again during and after the preprocessing to get more information about how the data is correlated. 

