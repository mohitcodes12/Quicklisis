import pandas as pd
from app.data_processing.profiling import profile_dataframe

def test_profile_dataframe():
    df = pd.DataFrame({"sales": [10, 20, None], "region": ["A", "A", "B"]})
    result = profile_dataframe(df)
    assert result["rows"] == 3
    assert result["columns"] == 2
    sales = next(x for x in result["columns_detail"] if x["name"] == "sales")
    assert sales["missing"] == 1
