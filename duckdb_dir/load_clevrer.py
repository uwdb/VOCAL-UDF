import duckdb
import os
import yaml

config = yaml.safe_load(
    open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
)
db_dir = config["db_dir"]

conn = duckdb.connect(database=os.path.join(db_dir, "annotations.duckdb"), read_only=False)

### Load Clevrer data ###
conn.execute("CREATE TABLE clevrer_objects (vid INT, fid INT, oid INT, oname VARCHAR, x1 INT, y1 INT, x2 INT, y2 INT);")
conn.execute("CREATE TABLE clevrer_relationships (vid INT, fid INT, rid INT, oid1 INT, rname varchar, oid2 INT);")
conn.execute("CREATE TABLE clevrer_attributes (vid INT, fid INT, oid INT, aname varchar);")
conn.execute("CREATE TABLE clevrer_relationship_predictions (vid INT, fid INT, rid INT, oid1 INT, rname varchar, oid2 INT);") # Same as clevrer_relationships
conn.execute("CREATE TABLE clevrer_attribute_predictions (vid INT, fid INT, oid INT, aname varchar);") # Same as clevrer_attributes

conn.execute("COPY clevrer_objects FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "clevrer_objects.csv")))
conn.execute("COPY clevrer_relationships FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "clevrer_relationships.csv")))
conn.execute("COPY clevrer_attributes FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "clevrer_attributes.csv")))
conn.execute("COPY clevrer_relationship_predictions FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "clevrer_relationships.csv"))) # Same as clevrer_relationships
conn.execute("COPY clevrer_attribute_predictions FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "clevrer_attributes.csv"))) # Same as clevrer_attributes