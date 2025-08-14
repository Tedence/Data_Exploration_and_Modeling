import os
from pathlib import Path
import numpy as np
import polars as pl
import gc
from nptdms import TdmsFile
from tqdm import tqdm


def read_tdms_file_improved(tdms_path, GMT=2, chunk_size=500000, downsample_factor=None, memory_threshold_mb=500):
    """
    Improved TDMS file reader with memory management and chunked processing.

    Parameters:
    -----------
    tdms_path : str or Path
        Path to the TDMS file
    GMT : int, default=2
        GMT offset in hours to apply to the Time column
    chunk_size : int, default=500000
        Number of samples to read at a time for large files
    downsample_factor : int, optional
        Factor by which to downsample the data (e.g., 2 means keep every 2nd sample)
    memory_threshold_mb : int, default=500
        Memory threshold in MB above which chunked reading is used

    Returns:
    --------
    pl.DataFrame or None
        Polars DataFrame containing the TDMS data, or None if reading fails
    """
    if not os.path.exists(tdms_path) or not str(tdms_path).lower().endswith('.tdms'):
        print(f"Invalid file path or not a TDMS file: {tdms_path}")
        return None

    try:
        print(f"Reading {tdms_path}...")

        # Read the TDMS file header first to get metadata
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

        # Get channel information and data length
        channels = list(group.channels())
        if not channels:
            print(f"Warning: No channels found in group")
            return None

        # Get the length of data from the first channel
        first_channel = channels[0]
        total_length = len(first_channel)
        print(f"Total data length: {total_length:,} samples")

        # Calculate memory requirements
        estimated_memory_mb = (total_length * len(channels) * 8) / (1024 * 1024)  # 8 bytes per float64
        print(f"Estimated memory requirement: {estimated_memory_mb:.1f} MB")

        if estimated_memory_mb > memory_threshold_mb:
            print("Large file detected, using chunked reading approach...")
            return _read_tdms_chunked(group, GMT, chunk_size, downsample_factor, total_length)
        else:
            # For smaller files, use the standard approach with optimizations
            return _read_tdms_standard(group, GMT, downsample_factor)

    except Exception as e:
        print(f"Error processing TDMS file {tdms_path}: {e}")
        return None


def _read_tdms_standard(group, GMT, downsample_factor=None):
    """Standard reading approach for smaller files"""
    try:
        data = {}

        for channel in group.channels():
            channel_data = channel[:]

            # Apply downsampling if specified
            if downsample_factor and downsample_factor > 1:
                channel_data = channel_data[::downsample_factor]

            # Use appropriate data type to save memory
            if channel.name.lower() == 'time':
                data[channel.name] = channel_data
            else:
                # Convert to float32 to save memory (vs default float64)
                data[channel.name] = channel_data.astype(np.float32)

        # Create a Polars DataFrame
        df = pl.DataFrame(data)

        # If the DataFrame has a 'Time' column, adjust it by GMT offset
        if 'Time' in df.columns:
            try:
                df = df.with_columns([pl.col('Time') + pl.duration(hours=GMT)])
                print(f"Adjusted 'Time' column by GMT+{GMT} hours")
            except Exception as e:
                print(f"Warning: Could not adjust 'Time' column: {e}")

        return df

    except Exception as e:
        print(f"Error in standard reading: {e}")
        return None


def _read_tdms_chunked(group, GMT, chunk_size, downsample_factor, total_length):
    """Chunked reading approach for large files"""
    try:
        channels = list(group.channels())

        if downsample_factor and downsample_factor > 1:
            print(f"Applying downsampling factor: {downsample_factor}")
            effective_chunk_size = chunk_size * downsample_factor
        else:
            effective_chunk_size = chunk_size

        # Process data in chunks
        chunk_dfs = []

        for start_idx in range(0, total_length, effective_chunk_size):
            end_idx = min(start_idx + effective_chunk_size, total_length)
            progress = (end_idx / total_length) * 100
            print(f"Processing chunk {start_idx:,} to {end_idx:,} ({progress:.1f}%)")

            chunk_data = {}

            for channel in channels:
                try:
                    channel_data = channel[start_idx:end_idx]

                    # Apply downsampling if specified
                    if downsample_factor and downsample_factor > 1:
                        channel_data = channel_data[::downsample_factor]

                    # Use appropriate data type
                    if channel.name.lower() == 'time':
                        chunk_data[channel.name] = channel_data
                    else:
                        chunk_data[channel.name] = channel_data.astype(np.float32)

                except Exception as e:
                    print(f"Warning: Error reading channel {channel.name} in chunk: {e}")
                    continue

            if chunk_data:  # Only create DataFrame if we have data
                chunk_df = pl.DataFrame(chunk_data)
                chunk_dfs.append(chunk_df)

            # Force garbage collection to free memory
            gc.collect()

        if not chunk_dfs:
            print("No valid chunks were processed")
            return None

        # Concatenate all chunks
        print("Combining chunks...")
        df = pl.concat(chunk_dfs, how='vertical')

        # Clear chunk list to free memory
        del chunk_dfs
        gc.collect()

        # If the DataFrame has a 'Time' column, adjust it by GMT offset
        if 'Time' in df.columns:
            try:
                df = df.with_columns([pl.col('Time') + pl.duration(hours=GMT)])
                print(f"Adjusted 'Time' column by GMT+{GMT} hours")
            except Exception as e:
                print(f"Warning: Could not adjust 'Time' column: {e}")

        print(f"Successfully processed large file with chunked reading")
        return df

    except Exception as e:
        print(f"Error in chunked reading: {e}")
        return None


def get_file_info(tdms_path):
    """
    Get basic information about a TDMS file without fully loading it.

    Parameters:
    -----------
    tdms_path : str or Path
        Path to the TDMS file

    Returns:
    --------
    dict
        Dictionary containing file information (groups, channels, length, etc.)
    """
    if not os.path.exists(tdms_path) or not str(tdms_path).lower().endswith('.tdms'):
        print(f"Invalid file path or not a TDMS file: {tdms_path}")
        return None

    try:
        tdms_file = TdmsFile.read(tdms_path)

        # Get file size
        file_size_mb = os.path.getsize(tdms_path) / (1024 * 1024)

        info = {
            'file_path': str(tdms_path),
            'file_size_mb': file_size_mb,
            'groups': {},
            'total_channels': 0
        }

        for group in tdms_file.groups():
            group_info = {
                'name': group.name,
                'channels': []
            }

            for channel in group.channels():
                channel_info = {
                    'name': channel.name,
                    'length': len(channel) if hasattr(channel, '__len__') else 'Unknown'
                }
                group_info['channels'].append(channel_info)
                info['total_channels'] += 1

            info['groups'][group.name] = group_info

        # Estimate memory requirements
        if info['total_channels'] > 0:
            # Get length from first channel found
            first_group = list(info['groups'].values())[0]
            if first_group['channels']:
                first_channel_length = first_group['channels'][0]['length']
                if isinstance(first_channel_length, int):
                    estimated_memory_mb = (first_channel_length * info['total_channels'] * 8) / (1024 * 1024)
                    info['estimated_memory_mb'] = estimated_memory_mb

        return info

    except Exception as e:
        print(f"Error getting file info for {tdms_path}: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Read TDMS files with improved memory management')
    parser.add_argument('tdms_file', help='Path to TDMS file')
    parser.add_argument('--gmt', type=int, default=2, help='GMT offset for time adjustment (default: 2)')
    parser.add_argument('--chunk-size', type=int, default=500000, help='Chunk size for large files (default: 500000)')
    parser.add_argument('--downsample', type=int, help='Downsampling factor (optional)')
    parser.add_argument('--memory-threshold', type=int, default=500, help='Memory threshold in MB (default: 500)')
    parser.add_argument('--info-only', action='store_true', help='Only show file information without loading data')
    parser.add_argument('--output', help='Output parquet file path (optional)')

    args = parser.parse_args()

    if args.info_only:
        # Show file information only
        info = get_file_info(args.tdms_file)
        if info:
            print(f"\n{'='*50}")
            print("TDMS FILE INFORMATION")
            print(f"{'='*50}")
            print(f"File: {info['file_path']}")
            print(f"Size: {info['file_size_mb']:.1f} MB")
            print(f"Total channels: {info['total_channels']}")
            if 'estimated_memory_mb' in info:
                print(f"Estimated memory: {info['estimated_memory_mb']:.1f} MB")

            for group_name, group_info in info['groups'].items():
                print(f"\nGroup: {group_name}")
                print(f"  Channels ({len(group_info['channels'])}):")
                for channel in group_info['channels']:
                    print(f"    - {channel['name']}: {channel['length']} samples")
    else:
        # Load and optionally save the data
        df = read_tdms_file_improved(
            args.tdms_file,
            GMT=args.gmt,
            chunk_size=args.chunk_size,
            downsample_factor=args.downsample,
            memory_threshold_mb=args.memory_threshold
        )

        if df is not None:
            print(f"\nSuccessfully loaded DataFrame with shape: {df.shape}")
            print(f"Columns: {df.columns}")

            if args.output:
                print(f"Saving to {args.output}...")
                df.write_parquet(args.output)
                print("Data saved successfully!")
        else:
            print("Failed to load TDMS file")
