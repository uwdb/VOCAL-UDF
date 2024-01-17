import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import os
from time import time
from pandas.testing import assert_frame_equal

def right_of(o1_x1: float, o1_x2: float, o2_x1: float, o2_x2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    return cx1 > cx2

def right_of_arrow_type(o1_x1: float, o1_x2: float, o2_x1: float, o2_x2: float) -> bool:
    cx1 = pc.divide(pc.add(o1_x1, o1_x2), 2)
    cx2 = pc.divide(pc.add(o2_x1, o2_x2), 2)
    return pc.greater(cx1, cx2)

# Create a DuckDB connection
# conn = duckdb.connect("/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=False)
conn = duckdb.connect()
conn.execute("CREATE TABLE Obj_clevrer (oid INT, vid INT, fid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float);")
conn.execute("COPY Obj_clevrer FROM '/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/obj_clevrer.csv' (FORMAT 'csv', delimiter ',', header 0);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid);")
conn.create_function("RightOf", right_of)
conn.create_function("RightOfArrow", right_of_arrow_type, type='arrow')
df = conn.sql("SELECT * FROM Obj_clevrer").to_view("df")
arrow_table = conn.execute("SELECT * FROM Obj_clevrer").arrow()
print("Start!")
_start = time()
native_df = conn.execute("""
    SELECT obj1_df.oid AS o1_oid, obj2_df.oid AS o2_oid, obj1_df.fid AS fid
    FROM df as obj1_df, df as obj2_df
    WHERE obj1_df.vid = obj2_df.vid AND obj1_df.fid = obj2_df.fid AND obj1_df.oid <> obj2_df.oid AND RightOf(obj1_df.x1, obj1_df.x2, obj2_df.x1, obj2_df.x2) = true
""").df()
_end = time()
print(f"Native UDF Time: {_end - _start}")

_start = time()
arrow_df = conn.execute("""
    SELECT obj1_df.oid AS o1_oid, obj2_df.oid AS o2_oid, obj1_df.fid AS fid
    FROM df as obj1_df, df as obj2_df
    WHERE obj1_df.vid = obj2_df.vid AND obj1_df.fid = obj2_df.fid AND obj1_df.oid <> obj2_df.oid AND RightOfArrow(obj1_df.x1, obj1_df.x2, obj2_df.x1, obj2_df.x2) = true
""").arrow()
_end = time()
print(f"Arrow UDF Time: {_end - _start}")
# print(len(native_df), len(arrow_df))

_start = time()
sql_query = conn.execute("""
    SELECT obj1_df.oid AS o1_oid, obj2_df.oid AS o2_oid, obj1_df.fid AS fid
    FROM df as obj1_df, df as obj2_df
    WHERE obj1_df.vid = obj2_df.vid AND obj1_df.fid = obj2_df.fid AND obj1_df.oid <> obj2_df.oid AND (obj1_df.x1 + obj1_df.x2) / 2 > (obj2_df.x1 + obj2_df.x2) / 2
""").df()
_end = time()
print(f"SQL Time: {_end - _start}")