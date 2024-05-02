import duckdb

conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)
feature_dir = "/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/features/clevrer_three_clips/attribute"
# attribute_features (vid INT, fid INT, o1_oid INT, feature float[])
res1 = conn.execute(f"""
    SELECT
        df.vid AS vid, df.fid AS fid, df.oid AS o1_oid, df.oname AS o1_oname, df.x1 AS o1_x1, df.y1 AS o1_y1, df.x2 AS o1_x2, df.y2 AS o1_y2,
        f.feature AS feature
    FROM clevrer_objects df
    JOIN '{feature_dir}/*.parquet' f
    ON f.vid = df.vid AND f.fid = df.fid AND f.o1_oid = df.oid AND f.o1_x1 = df.x1 AND f.o1_y1 = df.y1 AND f.o1_x2 = df.x2 AND f.o1_y2 = df.y2
    """).df()
print("done")

# res2 = conn.execute(f"""
#     SELECT COUNT(*)
#     FROM clevrer_objects
#     WHERE vid < 7500
#     GROUP BY vid, fid, oid
#     HAVING COUNT(*) > 2
#     """).df()
# print(res2)