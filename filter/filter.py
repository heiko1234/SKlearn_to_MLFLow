# Imports
import pandas as pd

# import numpy as np
import yaml

# from datetime import datetime
from pathlib import Path
import os


def read_configuration(configuration_file_path):
    """This function reads the Residencetime_setup.yaml
        file from the Source Code Folder"
    Arg:
        configuration_file_path[str]: path to that file

    Returns:
        [type]: yaml configuration used in this pipeline script
    """

    with open(configuration_file_path) as file:
        configuration = yaml.full_load(file)

    return configuration


def validate_filter(source_df, filters):
    parts = filters.split()
    if parts[0] in source_df.columns:
        return True
    else:
        print(f"{filters} is not valid")
        return False


def remove_spaces(alist):
    parts = alist.split(" ")
    parts = [item for item in parts if item != "" if item != " "]
    output = " ".join(parts)
    return output


def filter_condition(source_df, filters):

    for f in filters:

        output_df = []
        oparts = f.split("|")

        for opart in oparts:

            fparts = opart.split("&")
            each_filter = fparts[0]

            command = None
            output = None

            for each_filter in fparts:

                each_filter = remove_spaces(each_filter)

                if validate_filter(source_df=source_df, filters=each_filter):

                    parts = each_filter.split(" ")
                    parts = [
                        item for item in parts if item != "" if item != " "
                    ]

                    if len(parts) == 2:
                        command = 'source_df["{}"]{}'.format(
                            parts[0], parts[1]
                        )

                    if len(parts) == 3:
                        try:
                            if isinstance(float(parts[2]), float):
                                command = (
                                    'source_df["{}"] {} float("{}") '.format(
                                        parts[0], parts[1], parts[2]
                                    )
                                )

                        except ValueError:
                            if isinstance(str(parts[2]), str):
                                command = 'source_df["{}"] {} "{}"'.format(
                                    parts[0], parts[1], parts[2]
                                )

                else:
                    print(f"{each_filter} part of filter: {f} was not valid!")

                if output is not None and command is not None:
                    output = output + " & " + "(" + command + ")"

                elif output is None and command is not None:
                    output = "(" + command + ")"

            if output is not None:
                df = source_df.loc[eval(output)]

            else:
                df = source_df

            if df.shape[0] != 0:
                output_df.append(df)

            else:
                print(f"{opart} give no data")

        if len(output_df) == 1:
            source_df = output_df[0]
        else:
            source_df = pd.concat(output_df, axis=0)

        source_df = source_df.reset_index(drop=True)

        print(f"Apply numeric filter: {f}: {source_df.shape[0]}")

    return source_df


def filter_time(source_df, DateTime, filters):

    data = source_df

    if DateTime in source_df.columns:

        data[DateTime] = pd.to_datetime(data[DateTime], format="%Y-%m-%d")

        for i in filters:

            if i is not None:
                i = i.split(",")
                mask1 = data[DateTime] < i[0]
                mask2 = data[DateTime] > i[1]

                data = pd.concat([data.loc[mask1], data.loc[mask2]])
                data = data.reset_index(drop=True)
                print(f"Apply time excluding filter: {i}: {data.shape[0]}")
            else:
                continue
        return data

    else:
        print("No DateTime filter applied!")
        return data


def filtering_with_config(configuration):

    raw_data = pd.read_parquet(configuration["data_load"])

    return apply_filter(raw_data, configuration)


def apply_filter(data, configuration):

    filtered_data = filter_condition(
        source_df=data, filters=configuration["RawData_Filter"]
    )

    filtered_data = filter_time(
        source_df=filtered_data,
        DateTime=configuration["DateTime"][0],
        filters=configuration["Time_exclude_Filter"],
    )

    return filtered_data


def load_filter():

    path = Path(__file__).parent

    # configuration = read_configuration(configuration_file_path=
    #                     "/home/heiko/Repos/L-Dopa/filter/filter_config.yaml")
    configuration = read_configuration(
        configuration_file_path=os.path.join(path, "filter_config.yaml")
    )

    filtered_data = filtering_with_config(configuration=configuration)

    return filtered_data


def main():

    path = Path(__file__).parent

    configuration = read_configuration(
        configuration_file_path=os.path.join(path, "filter_config.yaml")
    )

    filtered_data = load_filter()

    print(f"save filtered data: {configuration['data_save']}")

    filtered_data.to_parquet(configuration["data_save"])


if __name__ == "__main__":

    main()
