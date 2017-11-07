from IV.data_extractor import create_PV_dataframe
import pandas as pd
import os
HERE = os.path.dirname(__file__)
test_data_loc = os.path.join(HERE, "test_data")

def test_create_PV_dataframe():
    df = create_PV_dataframe("test", "test",test_data_loc, light =True , force_analysis=True)
    assert isinstance(df, pd.DataFrame  )