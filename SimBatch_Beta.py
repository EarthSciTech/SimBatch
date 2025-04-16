"""
Code Description:
This script automates batch execution of reservoir simulations using tNavigator or Eclipse, processing multiple .DATA files. It:
1. Reads decision variables from a VARIABLES.INC file and locates all .DATA files in a specified directory.
2. Runs simulations sequentially or concurrently, parsing results from .rsm files for specified keywords.
3. Saves detailed reports, including end values and time-series data, to CSV files in a timestamped directory.
Key features:
- Supports CPU or GPU execution with configurable core usage and concurrent run limits.
- Converts scientific notation in results to general format for readability.
- Tracks simulation time and logs comprehensive outputs with units and decision variables.
Prerequisites:
- Input .DATA files must be compatible with tNavigator or Eclipse; VARIABLES.INC must define decision variables.
Dependencies: numpy, pandas, subprocess, concurrent.futures, datetime, csv, re (install via pip install numpy pandas).

Code by: Amirsaman Rezaeyan, amirsaman[dot]rezaeyan[at sign]gmail[dot]com, Calgary, Canada, Mon Jul  1 12:01:59 2024.
"""

import numpy as np  # Import NumPy for numerical operations and array handling
import re  # Import re for regular expression operations (e.g., parsing scientific notation)
import pandas as pd  # Import pandas for DataFrame operations and CSV/Excel handling
import os  # Import os module for file and directory operations
import subprocess  # Import subprocess to run external simulation executables
from concurrent.futures import ThreadPoolExecutor, as_completed  # Import for concurrent execution
import datetime  # Import datetime for timestamping and elapsed time calculations
import time  # Import time for handling delays during file access
import csv  # Import csv for writing CSV reports

# Define simulation run parameters
run_on_gpu = False  # Set to True for GPU simulations, False for CPU
num_cores = None  # Set to None to use all CPU cores, or an integer for specific core count
run_concurrent = False  # Set to True for concurrent .DATA file processing
max_concurrent_runs = 60  # Limit concurrent runs (None for no limit)

# Define the path to the simulator executable file
sim_executable_path = r"C:\Users\ar2052\AppData\Local\Programs\RFD\tNavigator\24.1\tNavigator-con-v24.1-2826-gba5e7f2.exe"  # Path to tNavigator executable

# Define the path to the folder containing the input file(s) for the simulations
input_file_path = r"D:\Amirsaman Rezaeyan\3- Model Realisations"  # Directory containing .DATA files

# Automatically find all .DATA files in the input_file_path
input_file_names = [f for f in os.listdir(input_file_path) if f.endswith('.DATA')]  # List all .DATA files

# Define Decision Variables
# Specify the folder and file name and create the full file path to the VARIABLES.INC file
variables_file_folder = r"D:\Amirsaman Rezaeyan\3- Model Realisations\INCLUDE\SCHEDULE"  # Directory for VARIABLES.INC
variables_file_name = "VARIABLES.INC"  # Name of variables file
variables_file = f"{variables_file_folder}/{variables_file_name}"  # Full path to VARIABLES.INC

# Define Keywords parsing objective functions and results
keywords = [
    "GUCEPUT", "FOPT", "FWIT", "FWPT", "FLPT", "FGPT", "FGIT", "FOIP", "FWIP", "FGIP",
    "GGLIT", "FOE", "FWCT", "FPR", "GUCEWTF", "GUCEWIP", "GUCEGLC", "GUCESPW", "GUCESPO",
    "GUCESPL", "GUCESPG", "GUCESPT", "GUCEDHW", "GUCEDHO", "GUCEDHL", "GUCEDHT", "GUCESTO",
    "GUCESTL", "GUCESTT", "GUCEGDT", "GUCEGTC", "GUCEPOT", "GUCEPGT", "GUCEPWT", "GUCEIWT"
]  # Keywords for parsing simulation results

# Function: Read the VARIABLES.INC file
def read_variables_file(variables_file):  # Read decision variables from VARIABLES.INC
    # Read the VARIABLES.INC file
    with open(variables_file, 'r') as file:  # Open file in read mode
        lines = file.readlines()  # Read all lines

    # Initialise the lists for decision variable names, base values, and units
    decision_variable_names = []  # List to store variable names
    base_values = []  # List to store base values
    decision_variable_units = []  # List to store units

    # Iterate over the lines in the VARIABLES.INC file
    for line in lines:  # Process each line
        # Ignore lines that do not start with a single quote (')
        if not line.strip().startswith("'"):  # Skip non-variable lines
            continue

        # Extract the decision variable name and values
        parts = line.split()  # Split line into parts
        decision_variable_name = parts[0][1:-1]  # Extract name, removing quotes
        base_value = float(parts[1])  # Convert base value to float
        decision_variable_unit = parts[6]  # Extract unit

        # Append the extracted values to the corresponding lists
        decision_variable_names.append(decision_variable_name)  # Add name
        base_values.append(base_value)  # Add base value
        decision_variable_units.append(decision_variable_unit)  # Add unit
        
    return decision_variable_names, base_values, decision_variable_units  # Return extracted data

# Function: Run simulations
def run_simulation(input_file_name, sim_executable_path, input_file_path, run_on_gpu, num_cores=None):  # Run a single simulation
    # Create the full input file path by combining the input_file_path and input_file_name
    full_input_file_path = f"{input_file_path}/{input_file_name}"  # Construct full path
    
    # Get the number of available CPU cores
    available_cores = os.cpu_count()  # Query available cores
    
    # If num_cores is not specified or exceeds available cores, use all available
    if num_cores is None or num_cores > available_cores:  # Check core specification
        num_cores = available_cores  # Default to max cores
    # Set up the command line arguments
    cmd_args = [sim_executable_path]  # Initialize with executable path
    
    # Run on GPU or CPU
    if run_on_gpu:  # Check if GPU is enabled
        cmd_args.append('--use-gpu')  # Add GPU flag
    else:
        cmd_args.extend(['--cpu-num', str(num_cores)])  # Add CPU core count
    
    # Generate the RSM file to collect results
    cmd_args.extend(['--ecl-rsm', '--use-rptrst-map-freq'])  # Enable RSM output
    # Direct to the DATA file path
    cmd_args.append(full_input_file_path)  # Add input file path
    
    # Run the simulation
    subprocess.run(cmd_args)  # Execute the simulation command

# Function: Convert scientific values to general values (written for parse_output_file)
def convert_sci_to_general(rsm_file_path):  # Convert scientific notation in RSM file
    # Open the rsm file and read its content
    with open(rsm_file_path, 'r') as file:  # Read RSM file
        content = file.read()  # Store content

    # Replace scientific notation with general notation
    def replacement(match):  # Define replacement function for regex
        # Get the original width of the matched scientific notation
        original_width = len(match.group(1))  # Get width of match
        # Get the sign of the exponent (either '+' or '-')
        exponent_sign = match.group(2)  # Extract exponent sign
        
        # Apply the appropriate format based on the sign of the exponent
        if exponent_sign == '-':  # Negative exponent case
            # Format with 6 decimal places
            converted_value = '{:.6f}'.format(float(match.group(1)))  # Convert to decimal
        else:
            # Use general format for positive/zero exponent
            converted_value = '{:g}'.format(float(match.group(1)))  # Convert to general
            
        # Return the converted value with the same width as the original
        return converted_value.ljust(original_width)  # Pad to original width

    # Replace all occurrences of scientific notation
    content = re.sub(r'([-+]?\d+\.\d*[eE]([-+]?)\d+)', replacement, content)  # Apply replacement

    # Write the updated content back to the rsm file
    with open(rsm_file_path, 'w') as file:  # Write back to file
        file.write(content)  # Save modified content

# Function: Try values only (written for parse_output_file)
def is_number(value):  # Check if a value is a number
    try:
        float(value)  # Attempt to convert to float
        return True  # Return True if successful
    except ValueError:
        return False  # Return False if conversion fails

# Function: Parse the results and units
def parse_output_file(rsm_file_path, keywords):  # Parse end values from RSM file
    # Convert values in scientific format to general format
    convert_sci_to_general(rsm_file_path)  # Normalize scientific notation
    
    # Parse the data from the file into a DataFrame
    df = pd.read_fwf(
        rsm_file_path,
        header=None,
        widths=[14, 13, 13, 13, 13, 13, 13, 13, 13, 12],
        comment='-',
        skip_blank_lines=True,
        dtype=None,
    )  # Read RSM file into DataFrame with fixed widths
    
    # Replace any infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Handle infinities
    
    # Identify the line number and column number where each keyword can be found
    keyword_indices = {}  # Dictionary to store keyword positions
    for keyword in keywords:  # Iterate over keywords
        for i, row in df.iterrows():  # Check each row
            if keyword in row.values:  # Look for keyword
                keyword_line_index = i  # Store row index
                keyword_part_number = row[row == keyword].index[0]  # Store column index
                keyword_indices[keyword] = (keyword_line_index, keyword_part_number)  # Save indices
                break
  
    # Extract the end values, units, and multipliers for each keyword
    end_value = []  # Initialize end value list
    multiplier = []  # Initialize multiplier list
    unit = {}  # Initialize unit dictionary
    result = {}  # Initialize result dictionary
    results = {}  # Initialize results dictionary
    Results = {}  # Initialize final Results dictionary
    units = {}  # Initialize units dictionary
    for keyword, (line_index, part_number) in keyword_indices.items():  # Process each keyword
        unit_line_index = line_index + 1  # Unit is on the next line
        unit = df.iloc[unit_line_index, part_number]  # Extract unit

        # Identify the line number where the end value is located
        end_value_line_index = None  # Initialize end value index
        for i in range(line_index + 1, len(df)):  # Search for end value
            row = df.iloc[i]  # Get row
            value = row.dropna().values[0]  # Get first non-NaN value
            if len(row.dropna()) == 1 and is_number(value) and float(value) == 1:  # Check for end marker
                end_value_line_index = i - 1  # Set end value to previous row
                break

        if end_value_line_index is None:  # Check if end value was found
            raise ValueError(f"Couldn't find the end value of the {keyword} column")  # Raise error if missing

        end_value = float(df.iloc[end_value_line_index, part_number])  # Extract end value
                
        # Identify the multiplier for the current keyword
        multiplier_line_index = line_index + 2  # Multiplier is two lines below
        multiplier_part = df.iloc[multiplier_line_index, part_number]  # Get multiplier text
        if multiplier_part and '*10**' in str(multiplier_part):  # Check for scientific notation
            multiplier = float(str(multiplier_part).split('*10**')[1])  # Extract exponent
        else:
            multiplier = 0  # Default to no multiplier
        
        # Calculate the result for the current keyword
        result = end_value * 10 ** multiplier  # Apply multiplier
        results[keyword] = (result, unit)  # Store result and unit
    
    # Separate the results and units
    Results = {k: results[k][0] for k in keywords}  # Extract numerical results
    units = {k: results[k][1] for k in keywords}  # Extract units
    
    return Results, units  # Return results and units

# Function: Parse all values with corresponding time steps and units
def parse_output_file_all(rsm_file_path, keywords):  # Parse time-series data from RSM file
    # Convert values in scientific format to general format
    convert_sci_to_general(rsm_file_path)  # Normalize scientific notation
    
    # Parse the data from the file into a DataFrame
    df = pd.read_fwf(
        rsm_file_path,
        header=None,
        widths=[14, 13, 13, 13, 13, 13, 13, 13, 13, 12],
        comment='-',
        skip_blank_lines=True,
        dtype=None,
    )  # Read RSM file into DataFrame
    
    # Replace any infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Handle infinities

    # Function to find the first value index for a given keyword
    def find_first_value_index(line_index, part_number):  # Find first valid value
        for i in range(line_index + 1, len(df)):  # Search from line after keyword
            value = df.iloc[i, part_number]  # Get value
            if not pd.isna(value) and is_number(value):  # Check if valid number
                try:
                    float(value)  # Verify float conversion
                    return i  # Return index
                except ValueError:
                    continue  # Skip invalid values
        return None  # Return None if no valid value found
    
    # Function to find the end value index for a given keyword
    def find_end_value_index(first_value_line_index):  # Find end of data
        for i in range(first_value_line_index, len(df)):  # Search from first value
            row = df.iloc[i]  # Get row
            value = row.dropna().values[0]  # Get first non-NaN value
            if len(row.dropna()) == 1 and is_number(value) and float(value) == 1:  # Check for end marker
                return i  # Return end index
        return None  # Return None if no end found

    # Define a list of time-related keywords to be handled
    time_keywords = ['DAY', 'MONTH', 'YEAR', 'TIME']  # Time-related keywords

    # Dictionary to store time-related data
    time_data = {}  # Initialize time data dictionary

    # Extract time-related data
    for time_keyword in time_keywords:  # Process each time keyword
        # Identify the line number for time steps
        time_step_index = None  # Initialize time step index
        for i, row in df.iterrows():  # Search rows
            if time_keyword in row.values:  # Look for keyword
                time_step_index = i  # Store row index
                time_step_part_number = row[row == time_keyword].index[0]  # Store column index
                break

        # Find the first value for the time steps
        first_time_value_line_index = find_first_value_index(time_step_index, time_step_part_number)  # Get first value index

        # Find end value for the time steps
        end_value_time_index = find_end_value_index(first_time_value_line_index)  # Get end index

        # Extract time steps
        time_steps = df.iloc[first_time_value_line_index:end_value_time_index, time_step_part_number].dropna().values.astype(float)  # Extract time steps

        # Extract time step unit
        time_step_unit = df.iloc[time_step_index + 1, time_step_part_number]  # Get unit

        # Store the time steps and unit in the dictionary
        time_data[time_keyword] = (time_steps, time_step_unit)  # Save time data
    
    # Identify the line number and column number where each keyword can be found
    keyword_indices = {}  # Initialize keyword indices dictionary   
    for keyword in keywords:  # Process each keyword
        for i, row in df.iterrows():  # Search rows
            if keyword in row.values:  # Look for keyword
                keyword_line_index = i  # Store row index
                keyword_part_number = row[row == keyword].index[0]  # Store column index
                keyword_indices[keyword] = (keyword_line_index, keyword_part_number)  # Save indices
                break
    
    # Extract all values, units, and multipliers for each keyword
    all_values = {}  # Initialize values dictionary
    units = {}  # Initialize units dictionary
    for keyword, (line_index, part_number) in keyword_indices.items():  # Process each keyword
        unit_line_index = line_index + 1  # Unit is on next line
        unit = df.iloc[unit_line_index, part_number]  # Extract unit
        units[keyword] = unit  # Store unit

        # Find the first value for the current keyword
        first_value_line_index = find_first_value_index(line_index, part_number)  # Get first value index
        
        # Find end value for the current keyword
        end_value_line_index = find_end_value_index(first_value_line_index)  # Get end index
        
        # Extract all values from the first value to the end value
        values = df.iloc[first_value_line_index:end_value_line_index, part_number].dropna().values.astype(float)  # Extract values
        
        # Apply multipliers to the values if necessary
        multiplier_part = df.iloc[line_index + 2, part_number]  # Get multiplier text
        if multiplier_part and '*10**' in str(multiplier_part):  # Check for scientific notation
            multiplier = float(str(multiplier_part).split('*10**')[1])  # Extract exponent
            values = values * (10 ** multiplier)  # Apply multiplier
        
        all_values[keyword] = values  # Store values
    
    return time_data, all_values, units  # Return time data, values, and units

# Function: Write all values with time steps to a CSV file
def write_all_values_to_csv(input_file_name, time_data, all_values, units, report_directory, decision_variable_names, decision_variable_units, base_values, current_time, start_time):  # Write time-series data to CSV
    # Create a CSV file for the input file name
    csv_file_name = f"{report_directory}/model_{input_file_name.replace('.DATA', '_alltime_report.csv')}"  # Define CSV file path
    
    # Open the main CSV file for writing
    with open(csv_file_name, 'w', newline='') as file:  # Open CSV in write mode
        writer = csv.writer(file)  # Initialize CSV writer
        
        # Write the header row
        header = ['DATE', 'TIME'] + list(all_values.keys()) + decision_variable_names + ['Date', 'Start Time', 'End Time', 'Elapsed Time']  # Define headers
        writer.writerow(header)  # Write header
        
        # Write the units row
        units_row = ['DAYS-MONTHS-YEARS', time_data['TIME'][1]] + [units[keyword] for keyword in all_values.keys()] + decision_variable_units + [''] * 4  # Define units row
        writer.writerow(units_row)  # Write units
        
        # Write the data rows
        for i in range(len(time_data['DAY'][0])):  # Iterate over time steps
            row = [
                f"{int(time_data['DAY'][0][i])}-{int(time_data['MONTH'][0][i])}-{int(time_data['YEAR'][0][i])}"  # Format date
            ] + [
                time_data['TIME'][0][i]  # Add time
            ] + [
                all_values[keyword][i] for keyword in all_values.keys()  # Add keyword values
            ] + [
                value for value in base_values  # Add decision variable values
            ] + [
                current_time.date(), start_time.time().strftime("%H:%M:%S"), datetime.datetime.now().time().strftime("%H:%M:%S"), str(datetime.datetime.now() - start_time).split(".")[0]  # Add timing info
            ]
            writer.writerow(row)  # Write data row
    
    # Write separate CSV files for each keyword
    for keyword in all_values.keys():  # Process each keyword
        keyword_csv_file_name = f"{report_directory}/keyword_{keyword}_report.csv"  # Define keyword CSV path
        
        # Check if the keyword CSV file already exists
        if os.path.exists(keyword_csv_file_name):  # If file exists
            # Load the existing keyword CSV file
            with open(keyword_csv_file_name, 'r') as file:  # Read existing CSV
                reader = csv.reader(file)  # Initialize reader
                existing_data = list(reader)  # Load data
            
            # Create the updated data
            updated_data = []  # Initialize updated data
            for i, row in enumerate(existing_data):  # Process each row
                if i == 0:
                    row.append(keyword)  # Add keyword to header
                elif i == 1:
                    row.append(units[keyword])  # Add unit
                elif i == 2:
                    row.append(input_file_name)  # Add file name
                else:
                    row.append(all_values[keyword][i-3])  # Add value
                updated_data.append(row)  # Store updated row
                
            # Write the updated data back to the keyword CSV file
            with open(keyword_csv_file_name, 'w', newline='') as file:  # Write updated CSV
                writer = csv.writer(file)  # Initialize writer
                writer.writerows(updated_data)  # Write rows
        else:
            # Create a new CSV file for the keyword
            with open(keyword_csv_file_name, 'w', newline='') as file:  # Create new CSV
                writer = csv.writer(file)  # Initialize writer
                
                # Write the header row
                writer.writerow(['DATE', 'TIME', keyword])  # Write header
                
                # Write the units row
                writer.writerow(['DAYS-MONTHS-YEARS', time_data['TIME'][1], units[keyword]])  # Write units
                
                # Write the file name row
                writer.writerow(['', '', input_file_name])  # Write file name
                
                # Write the data rows
                for i in range(len(time_data['DAY'][0])):  # Iterate over time steps
                    writer.writerow([
                        f"{int(time_data['DAY'][0][i])}-{int(time_data['MONTH'][0][i])}-{int(time_data['YEAR'][0][i])}",  # Write date
                        time_data['TIME'][0][i],  # Write time
                        all_values[keyword][i]  # Write value
                    ])     
               
# Function: Report table file for results, variables, and units
def report_table_endvalues(keywords, units, Results, input_file_name, decision_variable_names, decision_variable_units, base_values, report_file, row_out, results_out, start_time, current_time):  # Write end values to report
    # Create the units row for the report file
    if row_out == 1:  # Check if first row
        units_row = ['', ''] + [units[k] for k in keywords] + decision_variable_units  # Define units row
        while True:
            try:
                with open(report_file, 'a') as f:  # Open report file
                    f.write(','.join(units_row) + '\n')  # Write units
                break
            except IOError:
                print("Report file is currently open. Code is waiting before trying again. Close the report file immediately.")  # Log file access issue
                time.sleep(15)  # Wait before retrying
    
    # Write the rows, model names, results, decision variables, and time to the report file
    while True:
        try:
            with open(report_file, 'a') as f:  # Open report file
                f.write(f'{row_out},{input_file_name},')  # Write row number and model name
                f.write(','.join([f'{o:.6f}' for o in Results.values()]))  # Write results
                f.write(',')  # Add separator
                f.write(','.join([f'{p:.6f}' for p in base_values]))  # Write decision variables
                # Add date, time, start time, end time, and elapsed time
                end_time = datetime.datetime.now()  # Get current time
                elapsed_time = end_time - start_time  # Calculate elapsed time
                f.write(f',{current_time.date()},{start_time.time().strftime("%H:%M:%S")},{end_time.time().strftime("%H:%M:%S")},{str(elapsed_time).split(".")[0]}')  # Write timing info
                f.write('\n')  # End line
            break
        except IOError:
            print("Report file is currently open. Code is waiting before trying again. Close the report file immediately.")  # Log file access issue
            time.sleep(15)  # Wait before retrying

# Function: Obtain results
def obtain_results(variables_file, variables_file_folder, sim_executable_path, input_file_path, input_file_names, keywords, row_out, all_variants, report_directory, report_file, run_on_gpu, run_concurrent, max_concurrent_runs=None, num_cores=None):  # Main function to run and process simulations
    # Read the variables file to get base values
    decision_variable_names, base_values, decision_variable_units = read_variables_file(variables_file)  # Load decision variables
    
    # Get current date and time
    now = datetime.datetime.now()  # Get current timestamp
    # Format as string
    now_str = now.strftime("%Y-%m-%d_%H-%M")  # Format timestamp
    # Use this string in the directory name
    report_directory = f"{input_file_path}/REPORT_{now_str}"  # Define report directory
    if not os.path.exists(report_directory):  # Check if directory exists
        os.makedirs(report_directory)  # Create directory      
    
    # Define the filename for the report file storing iterations, objectives, results, and decision variables
    report_file_name = 'batch_endtime_report.csv'  # Name of main report file
    report_file = f"{report_directory}/{report_file_name}"  # Full path to report file    
    
    # Create the header row for the report file
    headers = ['No.', 'Model'] + keywords + decision_variable_names + ['Date', 'Start Time', 'End Time', 'Elapsed Time']  # Define headers
    with open(report_file, 'w') as f:  # Open report file
        f.write(','.join(headers) + '\n')  # Write headers             
    
    # Initialise the results matrix with zeros
    # Create a zero matrix named results_out
    num_files = len(input_file_names)  # Get number of input files
    num_keywords = len(keywords)  # Get number of keywords
    results_out = np.zeros((num_files, num_keywords))  # Initialize results matrix
    
    # Initialise lists to store results values for each input file
    results_list = []  # Initialize list for results
    
    # Run the simulation
    # Check if simulations should be run concurrently or sequentially
    if run_concurrent: # Run concurrently     
       # Initialise the timer and time
       current_time = datetime.datetime.now()  # Get start time
       start_time = datetime.datetime.now()  # Record start time
       
       # Print running status
       print('Batch job status: running', end='\r')  # Log status
       
       # If max_concurrent_runs is not set, run all simultaneously
       if max_concurrent_runs is None:  # Check concurrent run limit
          max_concurrent_runs = len(input_file_names)  # Set to total files
       
       # Create a ThreadPoolExecutor to manage concurrent simulation runs
       with ThreadPoolExecutor(max_workers=max_concurrent_runs) as executor:  # Initialize executor
            # Submit the simulation tasks to the executor
            future_to_file = {executor.submit(run_simulation, file, sim_executable_path, input_file_path, run_on_gpu, num_cores): file for file in input_file_names}  # Map futures to files
            
            # Iterate over the futures as they complete
            for future in as_completed(future_to_file):  # Process completed tasks
                input_file_name = future_to_file[future]  # Get file name
                
                try:
                    # Wait for the future to complete
                    future.result()  # Ensure simulation completed
                
                    # Construct the path to the .rsm file
                    rsm_file_path = f"{input_file_path}/RESULTS/{input_file_name.replace('.DATA', '')}/result.rsm"  # Define RSM path
                    
                    # Parse the results from the .rsm file
                    Results, units = parse_output_file(rsm_file_path, keywords)  # Extract end values
                    
                    # Parse all values with corresponding time steps
                    time_data, all_values, units = parse_output_file_all(rsm_file_path, keywords)  # Extract time-series data
                    
                    # Append the results to the results list
                    results_list.append((input_file_name, Results, units))  # Store results
                
                    # Write all values with time steps to CSV
                    write_all_values_to_csv(input_file_name, time_data, all_values, units, report_directory, decision_variable_names, decision_variable_units, base_values, current_time, start_time)  # Save time-series data
                
                except Exception as exc:
                       # Print an error message if an exception occurs
                       print(f'{input_file_name} generated an exception: {exc}')  # Log error
    
       for result in results_list:  # Process stored results
           # Call models-results-units list
           input_file_name, Results, units = result  # Unpack result
           
           # Call function reporting the table of results, variables, and units 
           report_table_endvalues(keywords, units, Results, input_file_name, decision_variable_names, decision_variable_units, base_values, report_file, row_out, results_out, start_time, current_time)  # Write end values
           
           # Extract all variants numbers
           all_variants.append(row_out)  # Track variant number
           # Increment the row counter
           row_out += 1  # Update row counter
           
       print(' completed.')  # Log completion
       
    else: # Run simulations sequentially         
         for input_file_name in input_file_names:  # Process each file
             # Initialise the timer and time
             current_time = datetime.datetime.now()  # Get start time
             start_time = datetime.datetime.now()  # Record start time
             
             # Print running status
             print(f'No.: {row_out}  Model: {input_file_name}  Status: running', end='\r')  # Log status
             
             # Call the run_simulation function
             run_simulation(input_file_name, sim_executable_path, input_file_path, run_on_gpu, num_cores)  # Run simulation
             
             # Construct the path to the .rsm file
             rsm_file_path = f"{input_file_path}/RESULTS/{input_file_name.replace('.DATA', '')}/result.rsm"  # Define RSM path
             
             # Extract the results and units from the .rsm file
             Results, units = parse_output_file(rsm_file_path, keywords)  # Extract end values
             
             # Parse all values with corresponding time steps
             time_data, all_values, units = parse_output_file_all(rsm_file_path, keywords)  # Extract time-series data
             
             # Write all values with time steps to CSV
             write_all_values_to_csv(input_file_name, time_data, all_values, units, report_directory, decision_variable_names, decision_variable_units, base_values, current_time, start_time)  # Save time-series data
             
             # Update results_out for the current simulation            
             results_out = [list(Results.values())]  # Update results matrix
             
             # Call function reporting the table of results, variables, and units 
             report_table_endvalues(keywords, units, Results, input_file_name, decision_variable_names, decision_variable_units, base_values, report_file, row_out, results_out, start_time, current_time)  # Write end values
             
             # Print completion status 
             print(' completed.')  # Log completion        
             
             # Extract all variants numbers
             all_variants.append(row_out)  # Track variant
             # Increment the row counter
             row_out += 1  # Update row counter
             
    return Results, units, all_variants  # Return final results

row_out = 1  # Initialize row counter
all_variants = []  # Initialize list for variant numbers
obtain_results(variables_file, variables_file_folder, sim_executable_path, input_file_path, input_file_names, keywords, row_out, all_variants, None, None, run_on_gpu, run_concurrent, max_concurrent_runs, num_cores)  # Run main function