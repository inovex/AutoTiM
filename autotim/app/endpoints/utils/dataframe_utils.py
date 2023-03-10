def convert_h2oframe_to_numeric(h2o_frame, training_columns):
    """Generate numeric h20 frame."""
    for column in training_columns:
        h2o_frame[column] = h2o_frame[column].asnumeric()
    return h2o_frame
