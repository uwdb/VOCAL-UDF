import duckdb

conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)

dataset = "clevrer"
n_videos = 10

attribute_domain = ["color_gray", "shape_cylinder", "material_metal", "location_right"]
relationship_domain = ["near", "right_of"]

attr_parameters = ','.join('?' for _ in attribute_domain)
rel_parameters = ','.join('?' for _ in relationship_domain)

def test_two_objects():
    conn.execute(f"""
        COPY (
            WITH obj_with_attrs AS (
                    SELECT
                        o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2,
                        COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS attributes
                    FROM {dataset}_objects o
                    LEFT OUTER JOIN {dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid AND a.aname = ANY([{attr_parameters}])
                    GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
                )
                , relationships_expanded AS (
                    SELECT
                        vid, fid, oid1, oid2,
                        COALESCE(ARRAY_AGG(rname) FILTER (WHERE rname = ANY([{rel_parameters}])), ARRAY[]::varchar[]) AS rnames,
                        ARRAY_AGG(rname) AS gt_rnames
                    FROM {dataset}_relationships
                    GROUP BY vid, fid, oid1, oid2
                )
                SELECT
                    o1.vid AS vid, o1.fid AS fid,
                    o1.oid AS o1_oid, o1.oname AS o1_oname, o1.x1 AS o1_x1, o1.y1 AS o1_y1, o1.x2 AS o1_x2, o1.y2 AS o1_y2, o1.attributes AS o1_anames,
                    o2.oid AS o2_oid, o2.oname AS o2_oname, o2.x1 AS o2_x1, o2.y1 AS o2_y1, o2.x2 AS o2_x2, o2.y2 AS o2_y2, o2.attributes AS o2_anames,
                    COALESCE(r1.rnames, ARRAY[]::varchar[]) AS o1_o2_rnames,
                    COALESCE(r2.rnames, ARRAY[]::varchar[]) AS o2_o1_rnames,
                    COALESCE(r1.gt_rnames, ARRAY[]::varchar[]) AS o1_o2_gt_rnames,
                    320 AS height, 480 AS width
                FROM obj_with_attrs o1
                JOIN obj_with_attrs o2 ON o1.vid = o2.vid AND o1.fid = o2.fid
                LEFT OUTER JOIN relationships_expanded r1 ON o1.vid = r1.vid AND o1.fid = r1.fid AND o1.oid = r1.oid1 AND o2.oid = r1.oid2
                LEFT OUTER JOIN relationships_expanded r2 ON o1.vid = r2.vid AND o1.fid = r2.fid AND o2.oid = r2.oid1 AND o1.oid = r2.oid2
                WHERE o1.oid <> o2.oid AND o1.vid < {n_videos}
                ORDER BY o1.vid, o1.fid, o1.oid, o2.oid
        ) TO '/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/clevrer_wide.csv' (FORMAT 'csv', delimiter ',', header)
    """, attribute_domain + relationship_domain)

def test_one_object():
    conn.execute(f"""
        COPY (
            SELECT
                o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
                o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
                COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
                320 AS height, 480 AS width
            FROM {dataset}_objects o
            LEFT OUTER JOIN {dataset}_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
            WHERE o.vid < {n_videos}
            GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2
            ORDER BY o.vid, o.fid, o.oid
        ) TO '/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/clevrer_wide.csv' (FORMAT 'csv', delimiter ',', header)
    """, attribute_domain)

if __name__ == "__main__":
    # test_one_object()
    test_two_objects()