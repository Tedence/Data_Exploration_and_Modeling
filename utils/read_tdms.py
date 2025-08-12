import os
from nptdms import TdmsFile
import polars as pl

def read_tdms_file_to_dataframe(tdms_path, GMT=2):
    """
    Read a TDMS file at the given path and convert it to a Polars DataFrame.
    Adjusts the 'Time' column by the specified GMT offset (default: GMT+2).

    Parameters:
    -----------
    tdms_path : str
        Path to the TDMS file
    GMT : int, default=2
        GMT offset in hours to apply to the Time column

    Returns:
    --------
    pl.DataFrame or None
        Polars DataFrame containing the TDMS data, or None if reading fails
    """
    if not os.path.exists(tdms_path) or not tdms_path.lower().endswith('.tdms'):
        print(f"Invalid file path or not a TDMS file: {tdms_path}")
        return None

    try:
        print(f"Reading {tdms_path}...")

        # Read the TDMS file
        tdms_file = TdmsFile.read(tdms_path)

        # Check if the 'Untitled' group exists
        if 'Untitled' not in tdms_file:
            # Get the available groups
            groups = list(tdms_file.groups())
            if not groups:
                print(f"Warning: No groups found in {tdms_path}")
                return None

            # Use the first available group
            group = groups[0]
            print(f"Using group '{group.name}' instead of 'Untitled'")
        else:
            group = tdms_file['Untitled']

        # Process each channel and store as columns in the DataFrame
        data = {}

        for channel in group.channels():
            data[channel.name] = channel[:]

        # Create a Polars DataFrame
        df = pl.DataFrame(data)

        # If the DataFrame has a 'Time' column, adjust it by GMT offset
        if 'Time' in df.columns:
            try:
                # Simply add GMT hours to the Time column
                df = df.with_columns([pl.col('Time') + pl.duration(hours=GMT)])
                print(f"Adjusted 'Time' column by GMT+{GMT} hours")
            except Exception as e:
                print(f"Warning: Could not adjust 'Time' column: {e}")

        print(f"Successfully processed {os.path.basename(tdms_path)}")
        return df

    except Exception as e:
        print(f"Error processing TDMS file {tdms_path}: {e}")
        return None