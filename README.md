# Disaster-Response-Pipeline-App
Data science nanodegree (udacity)

The messages send after a disaster can to help in labors of rescue, this project build an app web that contains an ML model, this model help to classifier a message in several categories of importance, and send it to appropriate disaster relief agency.
 
 # APP WEB.
 
- Home page

 ![Screenshot](image1.jpg) 

 ![alt text](https://raw.githubusercontent.com/dama207/Disaster-Response-Pipeline-App/blob/main/Images%20app/image1.jpg)


# Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/
