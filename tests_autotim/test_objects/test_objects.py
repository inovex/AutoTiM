"""Test objects."""
import os
import pandas as pd
import unittest
from random import random


def set_environ_for_testing(env_vars, set_with_random=False, set_with_none=True):
    for key in env_vars:
        if set_with_random:
            os.environ[key] = key + str(random())
        elif set_with_none:
            os.environ.pop(key, None)


def get_correct_test_df_from_env():
    return pd.DataFrame(data={
        os.getenv('COLUMN_ID'): [1, 2],
        os.getenv('COLUMN_VALUE'): [1, 2],
        os.getenv('COLUMN_KIND'): [1, 2]
    })


if __name__ == "__main__":
    unittest.main()
