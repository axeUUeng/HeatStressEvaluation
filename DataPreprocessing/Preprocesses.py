import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date, timedelta, datetime
# import datetime
import requests
import glob
import os


class MilkDataProcessor:
    def __init__(self) -> None:
        """
        Class to load GIGACOW data and correct corresponding MESAN file.

        """
        print(f"Hello {os.getlogin()}. Let's start")
        self.script_directory = os.path.dirname(os.path.realpath(__file__)) # the directory of this script
        self.parent_directory = os.path.dirname(self.script_directory) # its parent directory

        # directory of "raw" GIGACOW-data "/Data/Cowdata/rawGIGACOW/"
        self.rawGIGACOW_directory = os.path.join(self.parent_directory, 'Data', 'CowData', 'rawGIGACOW')
        os.makedirs(self.rawGIGACOW_directory, exist_ok=True)

        # directory to all MESAN files e.g. '/Data/WeatherData/rawMESAN'
        self.rawMESAN_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData', 'rawMESAN')  
        os.makedirs(self.rawMESAN_directory, exist_ok=True)
    
    def filter_data(self, startdate: str, enddate: str, farms: list=None) -> None:
        """
        Method of the MilkDataProcessor class to filter the data based on a start-date and a end-date.

        Args:
            - startdate (str): This is the startdate of type string e.g. '2022-01-01 00:00:00'.
            - enddate (str): This is the endate of type string e.g. '2023-11-13 23:00:00'.
            - farms (list): A list of farm pseudos. Deafult is None: then all farms are included
        Returns:
            - None
        """
        # directory to filtered GIGACOW data
        self.GIGACOW_directory = os.path.join(self.parent_directory, 'Data', 'CowData','GIGACOW')
        # Create the folder if it doesn't exists
        os.makedirs(self.GIGACOW_directory, exist_ok=True)
        print("\nFiltering GIGACOW based on dates...", end='', flush=True)
        pbar = tqdm(glob.glob(os.path.join(self.rawGIGACOW_directory, '*.csv')), desc = 'Filtering GIGACOW based on dates')
        for fname in pbar:
            filename = os.path.basename(fname) #Get the name of the .csv file e.g. MilkYield.csv
            pbar.set_description(f'Filtering {filename} based on dates')
            if filename == 'Cow.csv': #No dates in cow.csv so special case
                df = pd.read_csv(fname, delimiter=';', low_memory=False)
                # save a copy in new folder to make life easier
                output_file_path = os.path.join(self.GIGACOW_directory, f"{filename.replace('.csv', f'_filtered.csv')}")
                df.to_csv(output_file_path, index=False)
            else: # all other files has dates
                df = pd.read_csv(fname, delimiter=';', parse_dates=True, low_memory=False)
                if farms is not None: # if only need a subset of the farms
                    # assert that the requested farms exists in the data
                    assert all(farm in df['FarmName_Pseudo'].unique() for farm in farms), f"Not all farms exist in {filename}." 
                    df = df[df['FarmName_Pseudo'].isin(farms)].copy()
                
                # All datasets have slight different date and time naming conventions
                date_column = None
                time_column = None
                datetime_column = None
                    
                # Check for various date and time column naming conventions
                possible_date_columns = ['HealthEventDate', 'StartDate']
                possible_time_columns = ['StartTime']
                possible_datetime_columns = ['MilkingStartDateTime']
                
                for col in df.columns:
                    if any(word in col for word in possible_date_columns):
                        date_column = col
                    elif any(word in col for word in possible_time_columns):
                        time_column = col
                    elif any(word in col for word in possible_datetime_columns):
                        datetime_column = col
                
                # Filter rows based on the specified date range
                if datetime_column is not None:
                    df['DateTime'] = pd.to_datetime(df[datetime_column], errors='coerce')
                    filtered_df = df[(df['DateTime'] >= startdate) & (df['DateTime'] <= enddate)]
                elif date_column is not None and time_column is not None:
                    df['DateTime'] = pd.to_datetime(df[date_column] + ' ' + df[time_column], errors='coerce')
                    filtered_df = df[(df['DateTime'] >= startdate) & (df['DateTime'] <= enddate)]
                elif date_column is not None:
                    df['DateTime'] = pd.to_datetime(df[date_column], errors='coerce')
                    filtered_df = df[(df[date_column] >= startdate) & (df[date_column] <= enddate)]
                else:
                    raise ValueError(f"No valid date or datetime column found in {filename}") 
                
                # save each filtered file in /Data/CowData/GIGACOW/ as XXX_filtered.csv
                output_file_path = os.path.join(self.GIGACOW_directory, f"{filename.replace('.csv', f'_filtered.csv')}")
                filtered_df.to_csv(output_file_path, index=False)
        
        pbar.set_description('Filtering GIGACOW based on dates - Done')
        print("\rFiltering GIGACOW based on dates...Done!")



    def load_milk_data(self, farms: list = None) -> None:
        """
        Method of the MilkDataProcessor class to load and preproccess the MilkYield data.

        Args:
            - farms (list): A list of farm pseudos. Deafult is None: then all farms are included
        Returns:
            - None
        """
        print("\nLoading Milk Data...", end='', flush=True)
        # Read the milking data from '/Data/CowData/GIGACOW/MilkYield_filtered.csv'
        milk_data_directory = os.path.join(self.GIGACOW_directory, "MilkYield_filtered.csv")
        milk = pd.read_csv(milk_data_directory, delimiter=",")
        
        if farms is not None: # if only a subset of the farms are desired
            # assert that the requested farms exists in the data
            assert all(farm in milk['FarmName_Pseudo'].unique() for farm in farms), f"Not all farms exist in the Milkdata!."
            milk = milk[milk['FarmName_Pseudo'].isin(farms)].copy()
        
        # remove some columns
        milk_data = milk.drop(
            columns=[
                "Unnamed: 0",
                "LactationInfoSource",
                "SessionNumber",
                "OriginalFileSource",
                "dwh_factCowMilk_Id",
                "dwh_factCowMilkOther_Id",
            ]
        )
        # convert the 'TotalYield' to numeric format and replace commas with dots to handle decimal notation
        # and finally use 'errors='coerce' to replace any conversion errors with NaN values
        milk_data["TotalYield"] = pd.to_numeric(milk_data["TotalYield"].str.replace(",", "."), errors="coerce")

        milk_data["StartDate"] = pd.to_datetime(milk_data["StartDate"])  # convert to datetime format
        milk_data["DateTime"] = pd.to_datetime(milk_data["DateTime"])

        milk_data = milk_data.dropna()  # drop NaNs

        milk_data = milk_data.astype({"LactationNumber": "int"})  # Make the Lactationnumber into an integer

        # Read the cow data to get the correct breed
        cow_data_directory = os.path.join(self.GIGACOW_directory, "Cow_filtered.csv") 
        cows_df = pd.read_csv(cow_data_directory, delimiter=",")  

        # some duplicates exists - probably because of cows that have been moved/sold to different farms.
        # keep fist mention of each cow.
        cows_df = cows_df.drop_duplicates(subset=["SE_Number"], keep="first").reset_index(drop=True)

        # merge the milk data with the breed information
        self.milk_data = pd.merge(
            milk_data, cows_df[["SE_Number", "BreedName"]], on="SE_Number", how="left"
        )
        print("\rLoading Milk Data... Done!")

    def get_sick_data(self, farms: list=None) -> pd.DataFrame:
        """
        Method of the MilkDataProcessor class to load, preproccess and returns the DiagnosisTreatment data.

        Args:
            - farms (list): A list of farm pseudos. Deafult is None: then all farms are included
        Returns:
            - self.sick_data (Pandas.DataFrame): Dataframe of DiagnosisTreatment
        """
        # Read the Diagnosis data from '/Data/CowData/GIGACOW/DiagnosisTreatment_filtered.csv'
        sick_data_directory = os.path.join(self.GIGACOW_directory, "DiagnosisTreatment_filtered.csv")
        sick = pd.read_csv(sick_data_directory, delimiter=",")
        if farms is not None:
            assert all(farm in sick['FarmName_Pseudo'].unique() for farm in farms), f"Not all farms exist in the SickData!."
            sick = sick[sick['FarmName_Pseudo'].isin(farms)].copy()
        self.sick_data = sick.drop(
            columns=["Unnamed: 0", "Del_Cow_Id", "OriginalFileSource", "dwh_factCowHealth_Id",
                     ])
        self.sick_data["HealthEventDate"] = pd.to_datetime(self.sick_data["HealthEventDate"])
        self.sick_data["HealthEventOccurred"] = ~self.sick_data["HealthEventDate"].isna()
        return self.sick_data
    
    def weatherPreProcessing(self, points: list, start_date: datetime.date, end_date: datetime.date, farms=None) -> None:
        """
        Method of the MilkDataProcessor class to load, preproccess weather data.
        More specificly adds Global Irradiance and calcualtes THI_adj.
        Args:
            - points (list): List of dicts, with id: Farm Psuedo, lat: latitude and lon: longitude.
            - start_date (datetime.date): Start date
            - end_date (datetime.date): End date
            - farms (list): A list of farm pseudos. Deafult is None: then all farms are included
        Returns:
            None
        """
        param = 117 # [116, 117, 118, 119, 120, 121, 122]

        #"CIE UV irradiance", "Global irradiance", "Direct normal irradiance", "PAR", "Direct horizontal irradiance", "Diffuse irradiance"
        pname = "Global irradiance"
        interval = "hourly"
        # Create directory for processed weather data 'Data/WeatherData/MESAN/'
        self.MESAN_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData','MESAN')
        os.makedirs(self.MESAN_directory, exist_ok=True)
        print("\nAdding Global Irradiance and THI_adj...", end='', flush=True)
        
        if farms is not None:
            points = [point for point in points if point["id"] in farms]
        pbar = tqdm(points, desc = 'Adding Global Irradiance and THI_adj ...')
        for point in pbar:
            name = point["id"] #Farm Psuedo
            lat = point["lat"] #lat of farm
            lon = point["lon"] # lon of farm
            pbar.set_description(f'Adding Global Irradiance and THI_adj to {name}')
            
            fname = f"{name}_2022-2023.csv" # Name of 'raw' mesan file
            fpath = os.path.join(self.rawMESAN_directory, fname) # path to 'raw' mesan file

            if not os.path.exists(fpath): #If it doesn't exist, skip this farm
                continue
            # read it as a Pandas dataframe
            df = pd.read_csv(fpath, delimiter=';')

            #The following adds the STRÅNG parameter to the data frame   
            sDate = start_date.strftime('%Y-%m-%d')
            eDate = end_date.strftime('%Y-%m-%d')
            
            api_url = f"https://opendata-download-metanalys.smhi.se/api/category/strang1g/version/1/geotype/point/lon/{round(lon,6)}/lat/{round(lat,6)}/parameter/{param}/data.json?from={sDate}&to={eDate}&interval={interval}"
            
            #Runs the api call for a single parameter, puts the result in a dataframe and merges it with the main dataframe
            tf = pd.DataFrame(requests.get(api_url).json())
            tf = tf.rename(columns = {"value":pname, "date_time": "Tid"})
            tf['Tid'] = pd.to_datetime(tf['Tid']).dt.tz_localize(None)
            df['Tid'] = pd.to_datetime(df['Tid'])
            df = pd.merge(df, tf, on = "Tid")
            
            #THI calculation
            df["THI_adj"] = 4.51 + (0.8 * df["Temperatur"]) + (df["Relativ fuktighet"] * (df["Temperatur"] - 14.4)) + 46.4 - 1.992 * df["Vindhastighet"] + 0.0068 * df["Global irradiance"]
            

            #Writing the hourly data
            output_file_path = os.path.join(self.MESAN_directory, f"processed_data_{name}.csv")
            df.to_csv(output_file_path, index=False)
        print("\rAdding Global Irradiance and THI_adj... Done!")
            

    def add_weather_data(self, farms=None) -> None:
        """
        Method of the MilkDataProcessor class to combine weather data with milk data, and preprocess some more.
        Args:
            - farms (list): A list of farm pseudos. Deafult is None: then all farms are included
        Returns:
            - None
        """
        # Read the weather data from 'Data/WeatherData/MESAN/'
        print("\nAdding Weather Data...", end='', flush=True)
        files = os.listdir(self.MESAN_directory)  # a list of all mesan weather .csv files

        all_data = []  # here we will fill each farm's milk and weather data after merge
        all_farms = self.milk_data["FarmName_Pseudo"].unique() # list of farms
        if farms is not None: # check if we only want a subset of farms
            farms = set(farms)  # convert to a set for faster membership checks
            farms_to_process = [farm for farm in all_farms if farm in farms]
        else:
            farms_to_process = all_farms
        all_farms = tqdm(farms_to_process, desc="Merging weather and milk data", unit="farm")
        for farm_name in all_farms:  # iterate through all farms
            # farm XX

            # get the weather data for farm XX
            matching_files = [file
                              for file in files
                              if f"processed_data_{farm_name}.csv" == file.strip('"')]
            
            farm_data = self.milk_data[self.milk_data["FarmName_Pseudo"] == farm_name].copy()

            if (len(matching_files) == 1):  # Assert that we have find 1 and only 1 corresponding weather data file
                file_path = os.path.join(self.MESAN_directory, matching_files[0])
                temp_data = pd.read_csv(file_path)  # read the weather data

                temp_data["Tid"] = pd.to_datetime(temp_data["Tid"])  # maksure 'Tid' is datetime

                # Here we create the Heatwave columns. Defined by SMHI as 'Maxtemperature >= 25 degrees celsius for atleast 5 days in a row
                # in our definition; the 'HW' column will be 1 if the currect datapoint is experiencing a heatwave or if a heatwave has occured within
                # 1 week ago (7 days)
                temp_data.set_index("Tid", inplace=True)  # set the index to the timestamps
                temp_data = temp_data.sort_index()  # make sure they are orded by time
                max_temp_per_day = (temp_data["Temperatur"].resample("D").max())  # calculate the Max temp for each day

                max_temp_per_day = max_temp_per_day.reset_index()  # reset the index
                max_temp_per_day["HW"] = 0  # init a 'HW' column and set it to 0

                # The cumulative Heatwave is initially exactly like 'HW', but when a heatwave reaches > 5 days, an increment of 1 is added.
                # e.g. 5 days heatwave = [1, 1, 1, 1, 1] + 1 week of [1], but a cum_HW for 6, 7 and 8 days respectively will be:
                # cumulative heatwave (6 days of maxtemp >= 25 degrees) = [1, 1, 1, 1, 1, 2].
                # cumulative heatwave (7 days of maxtemp >= 25 degrees) = [1, 1, 1, 1, 1, 2, 3].
                # cumulative heatwave (8 days of maxtemp >= 25 degrees) = [1, 1, 1, 1, 1, 2, 3, 4].

                max_temp_per_day["cum_HW"] = 0.0

                # After the "cumulative heatwave" is over. an exponential decay is added to represent the lag of the heatwave.
                # The exponential decay is desgined to reach 0 after 7 days (1 week):
                # decay(x) = (A - 0.01*exp(x * 0.125 * log(100 * A))) where x is days after last day of heatwave, and A
                # is last value of incremental 'cum_HW'.
                # e.g. cumulative heatwave (7 days of maxtemp >= 25 degrees) = [1, 1, 1, 1, 1, 2, 3, (3 - 0.01*exp(1 * 0.125 * log(100 * 3)))
                #                                                                                  , (3 - 0.01*exp(2 * 0.125 * log(100 * 3)))
                #                                                                                  , (3 - 0.01*exp(3 * 0.125 * log(100 * 3)))
                #                                                                                  , ...
                #                                                                                  , (3 - 0.01*exp(7 * 0.125 * log(100 * 3))) ]

                # Iterate over groups of consecutive days with maximum temperature greater than or equal to 25 degrees Celsius
                for _, group_data in max_temp_per_day[
                    max_temp_per_day["Temperatur"] >= 25
                ].groupby((max_temp_per_day["Temperatur"] < 25).cumsum()):
                    if (len(group_data) >= 5):  # For each group, check if it contains at least 5 days
                        # If yes, mark the corresponding rows with 'HW' and 'cum_HW' to 1
                        max_temp_per_day.loc[group_data.index, "HW"] = 1
                        max_temp_per_day.loc[group_data.index, "cum_HW"] += 1.0
                        A = 1.0
                        if (len(group_data) > 5):  # If the group has more than 5 days, further adjust 'cum_HW' for the additional days
                            for i in group_data.index[5:]:
                                A += 1
                                max_temp_per_day.loc[i, "cum_HW"] = A

                        # Use an exponential decay formula to adjust 'cum_HW' values for the next 1 week after the heatwave
                        max_temp_per_day.loc[group_data.index[-1] + 1 : group_data.index[-1] + 7,"cum_HW"
                                             ] = (A - 0.01 * np.exp(np.arange(1, 8) * 0.125 * np.log(100 * A))
                        ).round(2)
                        # Set 'HW' to 1 for 1 week
                        max_temp_per_day.loc[
                            group_data.index[-1] + 1 : group_data.index[-1] + 7, "HW"] = 1

                temp_data["Tid"] = pd.to_datetime(temp_data.index)

                # Merge on the 'Tid' column
                temp_data = pd.merge(
                    temp_data,
                    max_temp_per_day[["Tid", "HW", "cum_HW"]],
                    how="left",
                    left_index=True,
                    right_on="Tid",
                )
                # forward-fill NaNs in the 'HW' and 'cum_HW' to propagate the last valid observation
                temp_data["HW"] = temp_data["HW"].ffill().astype(int)
                temp_data["cum_HW"] = temp_data["cum_HW"].ffill().astype(float)

                # remove the unnecessary Tid_y and Tid_x that is an artifact from the merge
                temp_data = temp_data.drop(columns=["Tid_y", "Tid_x"], axis=1)

                # reset the index after the merge
                temp_data = temp_data.reset_index(drop=True)

                # Extract 'DateTime' and format it to include only the date and hour
                farm_data.loc[:, "DateHour"] = farm_data["DateTime"].dt.strftime(
                    "%Y-%m-%d %H:00:00"
                )
                # Convert 'DateHour' back to datetime format
                farm_data["DateHour"] = pd.to_datetime(farm_data["DateHour"])

                # Merge milk data ('farm_data') with weather data ('temp_data') based on the 'DateHour' and 'Tid'.
                # The weather data has resolution 1hour, so we fill each hour of milk data with the same 1 hour weather data
                complete_data = pd.merge(
                    farm_data,
                    temp_data,
                    left_on="DateHour",
                    right_on="Tid",
                    how="inner",
                )
                complete_data.drop("DateHour", axis=1, inplace=True)  # Drop the 'datehour' column
                complete_data.drop("Tid", axis=1, inplace=True)  # as well as the 'Tid' column

                # Rename some columns to english
                complete_data.rename(columns = {'Relativ fuktighet':'Relative Humidity', 'Temperatur':'Temperature', 'Nederbörd':'Precipitation'}, inplace = True)
                all_data.append(complete_data)  # append to the list of datas

                # print(f"Completed data merge for {farm_name}") #Print a sanity check
            else:
                print(f"No matching file found for FarmName_Pseudo: {farm_name}")

        # merge all the datas from the all_data list of farmwise weather/milk data
        
        self.all_data = pd.concat(all_data)
        # drop some irelevant data
        self.all_data = self.all_data.drop(
            columns=["Snö", "Molnighet", "Sikt", "Del_Cow_Id", "Daggpunktstemperatur", "Vindhastighet", "Vindriktning", "Byvind", "Nederbördstyp", "Lufttryck", "Global irradiance",
            ])
        self.all_data['DateTime'] = pd.to_datetime(self.all_data['DateTime'], errors='coerce')
        self.all_data['StartTime'] = pd.to_datetime(self.all_data['StartTime'], format='%H:%M:%S', errors='coerce').dt.time
        self.all_data['StartDate'] = pd.to_datetime(self.all_data['StartDate'], errors='coerce')

        # remove outliers where the yield is below 2.5 kg and above 50 kg (for one milking instance, NOT DAILY)
        self.all_data = self.all_data[
            (self.all_data["TotalYield"] >= 2.5) & (self.all_data["TotalYield"] <= 50)
        ]
        
        # Save under 'Data/' as 'TheData.csv'
        output_file_path = os.path.join(self.parent_directory,'Data', f"TheData.csv")
        self.all_data.to_csv(output_file_path, index=False)
        print("\rAdding Weather Data... Done!")

    def preprocess(self, start_date, end_date, farms=None):
        """
        Method of the MilkDataProcessor class to preprocess the data.
        Args:
            - start_date (str): This is the startdate e.g. '2022-01-01 00:00:00'.
            - end_date (str): This is the enddate e.g. '2023-11-13 23:00:00'. 
            - farms (list): A list of farm pseudos. Deafult is None: then all farms are included
        Returns:
            - None
        """
        if farms is None:
            self.filter_data(start_date, end_date)
            coord_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData','Coordinates','Coordinates.csv')
            coord_df = pd.read_csv(coord_directory)
            coord_df = coord_df.dropna(subset=['FarmID'])
            coord_df[['lat', 'lon']] = coord_df['Koordinater'].str.split(', ', expand=True).astype(float)
            points = []
            for _, farm in coord_df.iterrows():
                input_point = {'id': farm['FarmID'], 'lat': farm['lat'], 'lon': farm['lon']}
                points.append(input_point)
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            startdate = start_datetime.date()
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            enddate = end_datetime.date()
            self.weatherPreProcessing(points, startdate, enddate)

            self.load_milk_data()
            self.add_weather_data()
            
        else:
            self.filter_data(start_date, end_date, farms)
            coord_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData','Coordinates','Coordinates.csv')
            coord_df = pd.read_csv(coord_directory)
            coord_df = coord_df.dropna(subset=['FarmID'])
            coord_df[['lat', 'lon']] = coord_df['Koordinater'].str.split(', ', expand=True).astype(float)
            points = []
            for _, farm in coord_df.iterrows():
                input_point = {'id': farm['FarmID'], 'lat': farm['lat'], 'lon': farm['lon']}
                points.append(input_point)
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            startdate = start_datetime.date()
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            enddate = end_datetime.date()
            self.weatherPreProcessing(points, startdate, enddate, farms)

            self.load_milk_data(farms=farms)
            self.add_weather_data(farms=farms)
        print('\nPreprocessing Done!')



def main():
    processor = MilkDataProcessor()
    start_date = '2022-01-01 00:00:00'
    end_date = '2023-11-13 23:00:00'
    processor.preprocess(start_date=start_date, end_date=end_date, farms = ['f454e660', 'a624fb9a'])


if __name__ == "__main__":
    main()
