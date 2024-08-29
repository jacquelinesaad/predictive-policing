"""
Name: Jacqueline Saad
Email: jacqueline.saad05@myhunter.cuny.edu
Resources: Similar Predictive Policing programs, Geopandas docs, Confusion Matrix docs.

Predictive Policing Data Science Project

Goal:
Analyze crime patterns in NYC and produce insights to be utilized in 
predictions for law enforcement & public safety.

Summary Statistics Functions: 
plot_crime_by_day_of_week:
Grouped bar charts showing crime counts by day of the week
for each borough. Color palette by rank is used for visual understanding.

create_crime_cat_bar_plot:
Bar plot showing the distribution of crime categories within each borough.
Allows for insight into the types of crimes more prevalent in each borough
and visualizes crime count within each borough.

create_time_series_plot_by_time:
Time series plot showing the distribution of crimes over time for each borough.
Allows for a comparison of crime patterns and what times
most crimes occur across different boroughs.

Map Graph Visualization:
create_choropleth_map:
Create a choropleth map of crime hotspots across NYC boroughs,
using crime occurrences, latitude, longitude and borough.
    
Visualizes the spatial distribution of crime rates.
    
Goals for Predictive Model Performance Plot:
Make predictions about where AND when violent crimes
are likely to occur to help proactive policing.
"""

import pandas as pd
import geopandas as gpd
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def load_data(file_name):
    '''
    Load the crime data from the CSV.
    '''

    data = pd.read_csv(file_name)
    # NYPD_Complaint_Data_Historic.csv

    # Drop rows with missing values.
    data = data.dropna()

    # Replace missing values in the 'BORO_NM' column with "Unknown".
    data['BORO_NM'] = data['BORO_NM'].replace('(null)', 'Unknown')

    return data

def calculate_crime_rates(data):
    """
    Calculate crime rates for each borough.
    """

    # Convert 'CMPLNT_FR_DT' to datetime format.
    data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], errors='coerce')

    # Drop rows with missing values in the 'CMPLNT_FR_DT' column.
    data = data.dropna(subset=['CMPLNT_FR_DT'])

    # Ensure 'CMPLNT_FR_DT' is a datetime type.
    data['CMPLNT_FR_DT'] = data['CMPLNT_FR_DT'].dt.to_pydatetime()

    # Define borough populations to be used for crime rates,
    # taken from recent census.
    borough_populations = {
        'BRONX': 1379946,
        'BROOKLYN': 2590516,
        'MANHATTAN': 1596273,
        'QUEENS': 2278029,
        'STATEN ISLAND': 491133
    }

    # Create a new column 'population' based on the dictionary.
    data['population'] = data['BORO_NM'].map(borough_populations)

    # Define a reference date for calculating time difference.
    reference_date = pd.to_datetime('2023-01-01')

    # Calculate the absolute difference between the crime date and the reference date.
    data['days_since_ref'] = (data['CMPLNT_FR_DT'] - reference_date).dt.days.abs()

    # Create a new column 'crime_rate' that indicates the number
    # of crimes per person in the borough.
    data['crime_rate'] = data['days_since_ref'] / (data['population'] + 1e-10)
    # Adding a small constant to avoid division by zero.

    # Calculate the total number of crimes for each borough.
    data['Number_of_Crimes'] = data.groupby('BORO_NM')['crime_rate'].transform('sum')

    return data

def extract_day_of_week(data):
    """
    Extract the day of the week from the 'CMPLNT_FR_DT' column
    and create a new column 'day_of_week'.
    """

    # Convert 'CMPLNT_FR_DT' to datetime format.
    data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], errors='coerce')

    # Extract the day of the week and map it to a string representation.
    data['day_of_week'] = data['CMPLNT_FR_DT'].dt.day_name()

    return data

def extract_hour(data):
    """
    Extract the hour from the 'CMPLNT_FR_TM' column
    and create a new column 'hour'.
    """
    # Convert 'CMPLNT_FR_DT' and 'CMPLNT_FR_TM' to datetime format.
    data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], errors='coerce')
    data['CMPLNT_FR_TM'] = pd.to_datetime(data['CMPLNT_FR_TM'], format='%H:%M:%S', errors='coerce')

    # Extract the hour of the day.
    data['hour'] = data['CMPLNT_FR_TM'].dt.hour

    return data

def define_violent_crime(data):
    """
    Define a new column 'violent_crime' based on 'LAW_CAT_CD'.
    """

    data['violent_crime'] = data['LAW_CAT_CD'].apply(lambda x: 1 if x == 'FELONY' else 0)
    return data


def explore_data(data):
    """
    Display basic information about the data.
    """

    print("First few rows of the data:")
    print(data.head())

    print("\nData information:")
    print(data.info())

    print("\nSummary statistics:")
    print(data.describe())

# Summary Statistics Plots:

def plot_crime_by_day_of_week(data):
    """
    Create separate grouped bar charts showing crime counts by day of the week for each borough.
    
    Allows for a comparison of crime counts by day of the week across different boroughs.
    Highlights days with more crimes with a darker color.
    """

    # Exclude rows with 'Unknown' borough.
    data_filtered = data[data['BORO_NM'] != 'Unknown']

    # Filter data for the year 2022.
    data_filtered_2022 = data_filtered[data_filtered['CMPLNT_FR_DT'].dt.year == 2022]

    # Group by 'BORO_NM' and 'day_of_week', and count the occurrences.
    crime_by_day_of_week = data_filtered_2022.groupby(['BORO_NM', 'day_of_week']).size().unstack()

    # Define the order of days of the week.
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Define color palettes for each borough.
    color_palettes = {
        'BRONX': sns.color_palette("OrRd", len(day_order)),
        'BROOKLYN': sns.color_palette("PuRd", len(day_order)),
        'MANHATTAN': sns.color_palette("YlOrRd", len(day_order)),
        'QUEENS': sns.color_palette("PuBuGn", len(day_order)),
        'STATEN ISLAND': sns.color_palette("GnBu", len(day_order))
    }

    # Set the order of days of the week.
    crime_by_day_of_week = crime_by_day_of_week[day_order]

    # Iterate through each borough.
    for borough in crime_by_day_of_week.index:
        # Get the ranking of crime counts for each day of the week.
        rank_order = crime_by_day_of_week.loc[borough].sort_values().index

        # Create a dictionary to map the day of the week to its rank.
        rank_mapping = {day: rank + 1 for rank, day in enumerate(rank_order)}

        # Map the rank to the corresponding color from the palette.
        color_palette = [color_palettes[borough][rank_mapping[day] - 1] for day in day_order]

        # Plot the bar chart with the adjusted color palette.
        plt.figure(figsize=(12, 6))
        sns.barplot(x=day_order, y=crime_by_day_of_week.loc[borough], palette=color_palette)

        plt.title(f'Total Crime by Day of the Week in {borough} - 2022')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Crimes')
        plt.show()

def create_crime_cat_bar_plot(data):
    """
    Create a bar plot showing the distribution of crime categories within each borough.
    
    Allows for insight into the types of crimes more prevalent in each borough.
    Unused in final project because of misleading crime rate due to population differences.
    """

    # Exclude rows with 'Unknown' borough.
    data_filtered = data[data['BORO_NM'] != 'Unknown']

    # Group by 'BORO_NM' and 'LAW_CAT_CD', and count the occurrences.
    crime_category_distribution = data_filtered.groupby(['BORO_NM', 'LAW_CAT_CD']).size().unstack()

    # Plotting the bar plot.
    crime_category_distribution.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Distribution of Crime Categories Within Each Borough')
    plt.xlabel('Borough')
    plt.ylabel('Number of Crimes')
    plt.legend(title='Crime Category')

    plt.xticks(rotation=0)  # Adjust the rotation angle for borough names for readability.
    plt.show()

def create_time_series_plot_by_time(data):
    """
    Create a time series plot showing the 
    distribution of crimes over time for each borough.
    
    Allows for a comparison of crime patterns and 
    what times most crimes occur across different boroughs.
    """

    # Exclude rows with 'Unknown' borough
    data_filtered = data[data['BORO_NM'] != 'Unknown']

    # Group by 'BORO_NM' and 'Hour', and count the occurrences
    crime_by_time = data_filtered.groupby(['BORO_NM', 'hour']).size().unstack()

    # Plotting the time series plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=crime_by_time.T, markers=True)

    # Format x-axis ticks with custom time format
    plt.xticks(range(3, 25, 3), ['12 am' if hour % 24 == 0 else '12 pm' if hour % 12 == 0
                                 else '{} am'.format(hour % 12) if hour < 12
                                 else '{} pm'.format(hour % 12) for hour in range(3, 25, 3)])

    # Set x-axis limits to start at 3 am and end at 12 am.
    plt.xlim(3, 24)

    plt.title('Crime Distribution by Time of Day for Each Borough')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Crimes')
    plt.legend(title='Borough', loc='upper right')

    plt.show()

# Map Graph Visualization:

def create_choropleth_map(data, geojson_path):
    """
    Create a choropleth map coloring each borough based on its crime count.
    """
    # Convert BORO_NM to title case to match the format in the GeoJSON file.
    data['BORO_NM'] = data['BORO_NM'].str.title()

    # Calculate 'crime_count' as the number of crimes in each borough.
    crime_counts = data['BORO_NM'].value_counts().reset_index()
    crime_counts.columns = ['BORO_NM', 'crime_count']

    # Create a Folium map centered on New York City.
    map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

    # Adding a Choropleth layer to the map.
    folium.Choropleth(
        geo_data=geojson_path,
        name='choropleth',
        data=crime_counts,
        columns=['BORO_NM', 'crime_count'],
        key_on='feature.properties.boro_name',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Crime Count'
    ).add_to(map)

    # Add layer control and save the map.
    folium.LayerControl().add_to(map)
    map.save('crime_count_map_borough.html')

    return map

# Predictive Model function:

def train_predictive_model(data):
    """
    Train a predictive model to forecast when and where violent crimes are likely to occur.
    """
    x = pd.get_dummies(data[['hour', 'BORO_NM']], drop_first=True)
    y = data['violent_crime']

    # Split the data into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Apply SMOTE to generate synthetic samples for balancing the dataset.
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    # Initialize and train the model on the resampled data.
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(x_train_resampled, y_train_resampled)

    # Predict on the test data.
    predictions = model.predict(x_test)

    # Evaluate the model using different metrics.
    print("Classification Report:\n", classification_report(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)
    return model, x_test, y_test, predictions

# Model Performance Plot:

def plot_model_performance(y_test, predictions):
    """
    Plot the model's performance showing Actual vs. Prediction.
    """
    conf_matrix = confusion_matrix(y_test, predictions)

    # Define labels for better interpretation.
    labels = ['Non-violent', 'Violent']

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Model Performance: Actual vs. Prediction')

    plt.show()

def main():
    """
    Main function.
    """

    # Replace file paths for csv & geojson as needed.
    data_path = r'C:\Users\Jackie\Downloads\cscilabs\NYPD_Complaint_Data_Historic.csv'
    geojson_path = r'C:\Users\Jackie\Downloads\cscilabs\Borough Boundaries.geojson'

    # Load data.
    data = load_data(data_path)

    # Calculate crime rates.
    data = calculate_crime_rates(data)

    # Explore data.
    explore_data(data)

    # Extract day of the week.
    data = extract_day_of_week(data)

    # Extract hour.
    data = extract_hour(data)

    # Define violent crime.
    data = define_violent_crime(data)

    # Plot crime counts by day of the week for each borough.
    plot_crime_by_day_of_week(data)

    # Create crime category bar plot.
    create_crime_cat_bar_plot(data)

    # Create time series plot.
    create_time_series_plot_by_time(data)

    # Create choropleth map.
    create_choropleth_map(data, geojson_path)

    # Predictive model and performance plot.
    model, x_test, y_test, predictions = train_predictive_model(data)
    plot_model_performance(y_test, predictions)

if __name__ == "__main__":
    main()
