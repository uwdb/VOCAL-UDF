import duckdb
import os
import yaml

config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)

# object_domain = conn.execute("select oname from vaw_objects group by oname order by count(*) desc limit 50").df()["oname"].to_list()
# print(object_domain)

gt_udf_names = ["black", "blue", "brown", "gray", "small", "metal", "long", "dark", "rounded", "orange", "white", "green", "large", "red", "wooden", "yellow", "tall", "silver", "standing", "round"]

for gt_udf_name in gt_udf_names:
    print("=====================================")
    print("gt_udf_name: ", gt_udf_name)
    sql = f"""
        SELECT
            o1.vid AS vid, o1.fid AS fid, o1.oid AS o1_oid, o1.oname AS o1_oname,
            o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
            COALESCE(ARRAY_AGG(a2.aname) FILTER (WHERE a2.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames_negative,
            m.height AS height, m.width AS width
        FROM vaw_objects o1
        LEFT OUTER JOIN vaw_attributes a ON o1.vid = a.vid AND o1.fid = a.fid AND o1.oid = a.oid
        LEFT OUTER JOIN vaw_attributes_negative a2 ON o1.vid = a2.vid AND o1.fid = a2.fid AND o1.oid = a2.oid
        LEFT OUTER JOIN vaw_metadata m ON o1.vid = m.vid AND o1.fid = m.fid
        GROUP BY o1.vid, o1.fid, o1.oid, o1.oname, o1.x1, o1.y1, o1.x2, o1.y2, m.height, m.width
        ORDER BY o1.vid, o1.fid, o1.oid, o1.x1, o1.y1, o1.x2, o1.y2
    """
    one_object_df = conn.execute(sql).df()

    print("len(one_object_df): ", len(one_object_df))

    df_filtered = one_object_df[(one_object_df["o1_gt_anames"].apply(lambda x: gt_udf_name in x)) | (one_object_df["o1_gt_anames_negative"].apply(lambda x: gt_udf_name in x))]

    print("len(df_filtered): ", len(df_filtered))

    df_pos = one_object_df[one_object_df["o1_gt_anames"].apply(lambda x: gt_udf_name in x)]
    print("len(df_pos): ", len(df_pos))

    df_neg = one_object_df[one_object_df["o1_gt_anames_negative"].apply(lambda x: gt_udf_name in x)]
    print("len(df_neg): ", len(df_neg))

