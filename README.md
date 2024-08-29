# predictive-policing

**Predictive Policing Data Science Project**

**Goal:** Analyze crime patterns in NYC and produce insights to assist law enforcement and public safety efforts through predictive modeling.
Description: This project aims to analyze historical crime data to identify patterns and visualize them through various statistical plots and maps. Additionally, a predictive model is developed to forecast when and where violent crimes are likely to occur, providing valuable insights for proactive policing.

**Libraries:**
pandas
geopandas
folium
seaborn
matplotlib
sklearn
imblearn

**Data Files:**
NYPD_Complaint_Data_Historic.csv (Crime data)
Borough Boundaries.geojson (NYC borough boundaries for map visualization)

**Data Loading:**
The data is loaded from a CSV file, cleaned by dropping missing values, and then processed to replace null borough names.
Crime Rate Calculation: Crime rates are calculated based on the population of each borough and the number of crimes reported.

**Functions:**
plot_crime_by_day_of_week: Creates grouped bar charts showing crime counts by day of the week for each borough.
create_crime_cat_bar_plot: Creates a bar plot showing the distribution of crime categories within each borough.
create_time_series_plot_by_time: Creates a time series plot showing the distribution of crimes over time for each borough.
**Map Visualization:**
create_choropleth_map: Creates a choropleth map of crime hotspots across NYC boroughs.

**Predictive Model Training:**
The model is trained using a Random Forest Classifier on features such as the hour of the crime and borough. SMOTE is applied to handle class imbalance.

**Predictive Model Evaluation:**
train_predictive_model: Trains the model and evaluates it using metrics like Mean Squared Error and a classification report.

**Performance Plotting:**
plot_model_performance: Visualizes the model's performance using a confusion matrix.

# **Usage Instructions**
Execute the main script to load the data, perform EDA, visualize the data, and train the predictive model.

**Results**
Visualization Outputs:
The project generates visualizations like crime distribution plots by day, time, and category, as well as a choropleth map.

Predictive Model Outputs:
The model predicts the likelihood of violent crimes occurring at specific times and locations, with performance metrics visualized for assessment.

