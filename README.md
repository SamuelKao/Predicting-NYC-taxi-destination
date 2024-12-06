# Where are you going?

## Data Download

* **Data Source:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
* **Data Description:** On the website you will find “Yellow Taxi Trip Records (PARQUET)” download links for every month of 2023, download these and place them on your google drive in a “CSE151GP” folder in a “data” subfolder (My Drive/CSE151GP/data/ is the relative path). The notebook will then mount to your drive and place this data into pandas dataframes once you run the program.
* **Taxi Zone Data source** https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc
* **Taxi Zone Data Description** Export the data into a tsv file and make sure it has the name "taxi_zones.tsv". Then place the tsv file in the same data folder as with the parquet files of the data. The notebook will then be able to read this file as well once you have mounted your drive.

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
## Abstract
If you enter one of the famous yellow Taxis in New York City (NYC), the first question is “where are you going?” But what if the driver already had a list of the top 5 most common places to go, based on where they are and what time it is or other useful data that is known at that time. This could save loads of time and effort for everyone involved. The Taxi and Limousine Commission (TLC) have kept data on all cabs including pick-up and drop-off locations, timestamps, trip-distance, fare amount, tolls, passenger count and more. Using some of this data, this project aims to train a machine learning model to find the end location for a given taxi ride and possibly what the fare amount would be. All the common pick-up and drop-off locations are already encoded by TLC, making this a classification problem with a finite number of locations all over NYC. During 2023 a total of 38 million rides were documented, giving lots of data to use for the model. 

## Introduction
Imagine stepping into one of New York City's iconic yellow taxis and having the driver already know your most likely destination based on the time, location, and other contextual data. Predicting NYC taxi destinations was chosen not just for its technical challenge but because it offers tangible benefits that extend far beyond convenience. With over 38 million taxi rides documented in 2023, the dataset provided by the Taxi Limousine Commission (TLC) offers a unique opportunity to explore real-world applications of machine learning on an unprecedented scale.

What makes this project exciting is its potential to revolutionize urban transportation. A highly accurate predictive model could reduce wait times for passengers, improve route planning for drivers, and even inform city infrastructure decisions by highlighting high-demand areas. 

## Methods
### Preprocessing

#### Data Description
The data consists of a 12 parquet files with data from all cabs in New York in 2023. We then load these files into data frames, each represententing a different month of the year, and in total it is around 38 million data points. Every datapoint has 19 features of different kinds of formats ranging from datetime, int, float and object, meaning that some features need to be encoded in order to be useful. The data is retrieved from The Taxi & Limousine Commission (TLC) and includes some data points with missing features and outliers and therefore the data need to be cleaned before it can be used. 

#### Relevant data
Our model is meant to be used before the cab starts riding, meaning that a lot of features are "unknown". The features we can use are then pickup time(tpep_pickup_datetime), passenger count and pickup location(PULocationID), while dropoff location(DOLocationID) is the feature we want to predict. 

tpep_pickup_datetime is formatted as a datetime64 and must be encoded to effectively incorporate date and time information into our model. First, we plan to apply **cyclical encoding** to both day and time. Specifically, we’ll convert the datetime columns into two features: one representing the number of days since the start of the year and another representing seconds since the start of the day. By then applying sine and cosine transformations to these values, we can capture the cyclical nature of time, allowing the model to understand that day 365 is close to day 1 and that 23:59:59 is close to 00:00:00. The resulting values will be scaled between -1 and 1 for all four columns. 
We also plan to encode the **day of the week** using ordinal encoding (e.g., Monday=0, Tuesday=1, etc.), which preserves the natural order of days while scaling from 0 to 6 as integer values. This will help the model interpret weekday patterns directly. In addition, we intend to include **holiday features**. Specifically, we’ll create indicators for whether a day is the day before a holiday, an actual holiday, or the day after a holiday. If a day meets any of these criteria, we will note the specific holiday. At this stage, our tentative plan is to encode these indicators as true/false values, as the time of year may already provide sufficient context regarding the holiday itself. This means that the resulting scale for holiday features will remain minimal.

The feature passenger_count might not be a strong predictor on its own but it could offer insights into the destination type. For example, taxis carrying more passengers may be heading towards popular locations that accommodate larger groups, such as tourist spots, hotels, airports, or event venues. Therefore we plan to feed it into the model. 
For PULocationID and DOLocationID the locations in the dataset are encoded with a location id 1 to 262 that can be mapped to a location in New York and this data is found separately in an encoding sheet. The location data consists of Shape_Leng, the_geom, Shape_Area, zone, LocationID, and borough. Zone is a string with the name of the location, useful to understand the data and borough is a string with the district of the location. LocationID is the location id that we will map together with our main data. Shape_Leng, the_geom, Shape_Area together describe a polygon of the location area in NY that is correlated to the specific location id. This data consists only of 262 entries with location 56 repeated twice and 103 trice, while locationID 104, 105, 57 is not included at all. Looking at other maps of the zone codes in NYC, it is apparent that this is just a mistake in the location ID data that can easily be fixed. 

Since the model is a classification model, we plan to encode the drop off location ids with one hot encoding. This will also help us later to get the 5 most probable drop off locations. For the pickup location however, we want to encode the location to make sure the model understands which locations are close together. To do this we will use ordinal encoding based on the coordinates from the polygon description in the encoding sheet. For example we could sort them north to south or maybe both north to south and west to east to give the model more features to work with. 

#### Data distribution
For the relevant data we want to take a look on how it is distributed. 

The pick-up and drop-off locations are distributed as seen in the figures below. It is clear that some locations are more popular than others meaning that we have unbalanced dataset. It is also not possible to conclude anything about which locations are close together and not indicating the importance of preprocessing the locations. 

<img width="887" alt="image" src="https://github.com/user-attachments/assets/3f64fd34-31ba-4fd9-b046-423ba7bac38e">

The timestamps of the taxi rides are distributed over the day and can be seen in the figure below. We can see that there are more rides during the day than in the night. This is just a small sample of all of the data but we can see that it follows a reasonable pattern which is good. 

![image](https://github.com/user-attachments/assets/bbdda38e-5572-4ef6-8fa2-c886c16f5f68)

Similarly the data is distributed over the entire year in the figure seen below where we have plotted a small sample of the data. We expect the data to be evenly distributed over the year and we can see that it usually does. However we can identify a smaller frequency in september and even after switching samples we still get the same result. Investigating this further we did not find any special event happening in New York for this day, but since we still have a lot of data we dont think it will influence the model. 

![image](https://github.com/user-attachments/assets/3956a34a-cea4-4d5a-bf00-01c240939d2f)

For the Passenger_count, the distribution is shown in the figure below. We can see that there is usually only one passenger but sometimes there is more. There is also a few rides with zero passengers and we can assume that this is what happens if the driver fails to report it. 

<img width="450" alt="image" src="https://github.com/user-attachments/assets/0446a7b6-2721-4f6a-a39f-4538ce81281f">


#### Missing data
Missing data have been found for the features passenger_count, RatecodeID, store_and_fwd_flag, congestion_surcharge and Airport_fee. Since the only feature we will use out of these is passenger_count we only need to handle this one. We plan to use median imputation which for this data is 1 passenger, this will be both for when the data is null and 0 since 0 passengers would not count as a cab ride. 

#### Outliers
While exploring the data, it was noticed that there are some dates that there were about 100 rides not in the year of 2023, these data points will be dropped. There are also 2475 sets of passengers who were dropped off before they were picked up, and these data points will also be dropped. The same goes for the 30051 rides longer than 10 hours, since we believe that these stem from taxi drivers forgetting to turn off the taximeter, and 10 hours is the maximum amount of time a taxi driver is allowed to work within a 24 hour window. We also consider trips with a trip distance that is unrealistic short to be outliers and remove those. 
Since the data we will use needs to be encoded, it is not possible to create any correlation or scatterplot of the relevant data at this point. There is a pairplot in the code, but this will need to be addressed again during and after the preprocessing to get more information about how the data is correlated. 

#### Final preprocessing
The final preprocessing included processing the tpep_pickup_datetime to get 'year', 'sine_days_since_year_began', 'cosine_days_since_year_began', 'sine_seconds_since_day_began', 'cos_seconds_since_day_began', 'day_of_week', 'holiday_today', 'DateOffset', 'day_after_holiday', 'holiday_yesterday', 'day_before_holiday' and 'holiday_tomorrow'. The PULocationID was processed by getting the mean x and y coordiate from the multipolygon object based on the location id in the encoding sheet to get 'coordx' and 'coordy'. For the missing passengercount attribute it is filled with the median value in the data. 

### Evaluation
The goal of the model is to predict the 5 most probable dropoff locations. To evaluate the model we therefore consider it a success if the true dropoff location is any of the predicted top 5 most probable dropoff location. The accuracy of the model is therefore the amounth of times the model predicted the correct location within the top 5 devided by the length of the testset. 

### Model 1
When training the first model a sample of the data was used to get an understanding what models work best on the data as efficiently as possible. The choosen model is a multinomial logistic regression model to predict the one hot encoded dropoff location. The sample used includes around 38000 entries that were split into 80% training data and 20% test data.

The model is described as the following formula where we are training the beta values.

![image](https://github.com/user-attachments/assets/2155cb95-af2e-4d4f-b479-a2a1af48dd12)


#### Process of first model
The first model had an accuracy of 12.38% on the test data and 14.01% on the train data, using the definition of accuracy as described. Comparing this to a random classified that would have an accuracy of 5/262 = 1.9%, our model is quite better than the random one. We can also conclude that the multinomial logistic regression is very computionally heavy and takes very long to train without getting extrodinary result. It does not handle a high number of features well, and we suspect that this is the main reason for the model to train slowly. 
In the fitting graph, the model is underfitting since the training MSE has not improved more than the test MSE. Therefore we know that the model needs futher training or a another model should be used. Below the fitting graph is shown.

![image](https://github.com/user-attachments/assets/1bd60787-a249-46e5-9354-841cc5d8a0b3)

For the next model we are thinking about training a neural network since that is scaleable to bigger datasets. The multinomial logistic regression takes very long even for the small sample we used and since we have a large dataset we want to use a model that can handle all of that to get a better model. For the next model we therfore should also be able to use more of the data which will probably lead to better accuracy as well. 

### Model 2 
The second(and technically third) model is a Multi Layer Perceptron (MLP) Neural network that also predicts a probability for every output class. First, sk-learns MLP model was used. This model was relatively slow which was balanced out by using less data. After switching to our own implementation of an MLP with pytorch, we got a faster model and could therefor also use more data for training the model. 

#### Result of second model
The accuracy, as described above, resulted in 27.1% for testing data and 27.0% for training data. The model was trained using 1% of our data.

#### Conclusion of second model
The model is better than the first one, and it especially allowed us to use more of our data while also being faster to train. In the fitting graph we are still underfitting since the training and test accuracy is both slowly still improving for every training epoch when we stop training. This leads us to believe that we can improve our model with more training and if we use more data. 
For tuning the model, we focused on finding the best kind of model which landed us in our own version with pytorch after trying sk learns version.
The main tuning we did was for the model size, concluding that a bigger model was better, but we have yet to find an optimal size.
To improve the model we therefore also plan to do some more hyperparameter tuning, for example the learning rate, batch size and model size. 

For the final model we are happy with using a MLP model and together with some hyperparameter tuning and training with more data we are hopeful to achieve a slightly better version than the second model. 

## Results

### Model 1
The first model had an accuracy of 12.38% on the test data and 14.01% on the train data, using the definition of accuracy as described in the evaluation section. 

### Model 2
The accuracy, as described in the evaluation section, resulted in 27.1% for testing data and 27.0% for training data. The model was trained using 1% of our data.

### Final Model

## Discussions

### The task 
There are many possibilities for where someone might want to go in New York and it is therefore clear that the task of predicting where someone is going is a very difficult task. 

### Model 1


### Model 2


### Final Model

## Conclusions

## Statement of contribution

Axel Orrhede: title: Contribution
Rebecka Eldh: title: Contribution
Samuel Kao: title: Contribution
