import os

import pandas as pd

def file_extension(path: str, extension: str) -> bool:
    # Split the extension from the path and normalise it to lowercase.
    ext_current = os.path.splitext(path)[-1].lower()

    return ext_current == extension

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_single_file_input_format(request, allowed_extensions):
    if len(request.files) != 1:
        return None
    file = request.files[list(request.files.keys())[0]]

    if allowed_file(filename=file.filename, allowed_extensions=allowed_extensions):
        return file
    return None

def read_timeseries_from_file(file):
    try:
        if file_extension(path=file.filename, extension='.csv'):
            timeseries = pd.read_csv(file)
        elif file_extension(path=file.filename, extension='.json'):
            timeseries = pd.read_json(file)
    except (AttributeError, ValueError):
        timeseries = None
    return timeseries
