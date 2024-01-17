import duckdb
import os
import pandas as pd
from tqdm import tqdm
from pandas.testing import assert_frame_equal

# Remove file if exists
if os.path.exists('test.duckdb'):
    os.remove('test.duckdb')

conn = duckdb.connect(database='test.duckdb', read_only=False)
conn.execute("CREATE TABLE Obj_clevrer (oid INT, vid INT, fid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)")
conn.execute("COPY Obj_clevrer FROM '/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/obj_clevrer.csv' (FORMAT 'csv', DELIMITER ',', HEADER 0);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid);")

query_results = []
for vid in tqdm(range(500)):
    df1 = conn.execute(f"""
        SELECT oid as o0_oid, fid
        FROM Obj_clevrer
        WHERE color = 'red' AND vid = {vid}
    """).df()
    df2 = conn.execute(f"""
        SELECT oid as o0_oid, fid
        FROM Obj_clevrer
        WHERE material = 'rubber' AND vid = {vid}
    """).df()
    graph_df1 = conn.execute(f"SELECT * FROM df1 natural join df2").df()
    if len(graph_df1):
        query_results.append(vid)

ground_truth = [1, 2, 3, 6, 14, 21, 23, 25, 29, 30, 32, 33, 40, 41, 44, 50, 51, 54, 63, 71, 74, 75, 78, 79, 83, 84, 90, 91, 94, 95, 97, 98, 100, 105, 107, 109, 111, 114, 115, 117, 125, 126, 128, 132, 138, 139, 144, 146, 149, 151, 154, 156, 163, 165, 169, 171, 177, 180, 183, 185, 186, 189, 191, 193, 197, 201, 202, 205, 206, 207, 209, 214, 217, 221, 222, 223, 235, 241, 244, 248, 249, 250, 252, 254, 257, 261, 266, 268, 270, 271, 276, 287, 291, 292, 299, 308, 314, 315, 317, 326, 342, 344, 346, 347, 349, 350, 351, 355, 364, 366, 367, 368, 369, 372, 380, 384, 392, 394, 399, 400, 402, 406, 407, 410, 412, 414, 416, 418, 419, 420, 424, 428, 431, 433, 435, 438, 442, 448, 450, 452, 456, 462, 463, 464, 476, 477, 479, 480, 484, 485, 486, 489, 490, 493]

diff1 = sorted(list(set(query_results) - set(ground_truth)))
print("diff1", diff1)

if len(diff1):
    # This gives incorrect results
    df1 = conn.execute("select * from Obj_clevrer where vid = {} and fid = 0".format(diff1[0])).df()
    print(df1.to_string())

    # This gives correct results
    df2 = conn.execute("select * from Obj_clevrer").df()
    print(df2.loc[(df2['vid'] == diff1[0]) & (df2['fid'] == 0)].to_string())
