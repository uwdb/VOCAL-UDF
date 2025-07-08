import duckdb
import os
import yaml

config = yaml.safe_load(
    open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r")
)
db_dir = config["db_dir"]

conn = duckdb.connect(database=os.path.join(db_dir, "annotations.duckdb"), read_only=False)

### Load CityFlow data ###
conn.execute("CREATE TABLE cityflow_metadata (vname varchar, vid INT, fid INT, width INT, height INT);")
conn.execute("CREATE TABLE cityflow_objects (vid INT, fid INT, oid INT, oname varchar, x1 int, y1 int, x2 int, y2 int);")
conn.execute("CREATE TABLE cityflow_relationships (vid INT, fid INT, rid INT, oid1 INT, rname varchar, oid2 int);")
conn.execute("CREATE TABLE cityflow_attributes (vid INT, fid INT, oid INT, aname varchar);")
conn.execute("CREATE TABLE cityflow_relationship_predictions (vid INT, fid INT, rid INT, oid1 INT, rname varchar, oid2 int);") # Same as cityflow_relationships
conn.execute("CREATE TABLE cityflow_attribute_predictions (vid INT, fid INT, oid INT, aname varchar);")

conn.execute("COPY cityflow_metadata FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "cityflow_metadata.csv")))
conn.execute("COPY cityflow_objects FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "cityflow_objects.csv")))
conn.execute("COPY cityflow_relationships FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "cityflow_spatial_relationships.csv")))
conn.execute("COPY cityflow_attributes FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "cityflow_attributes.csv")))
conn.execute("COPY cityflow_relationship_predictions FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "cityflow_spatial_relationships.csv"))) # Same as cityflow_relationships
conn.execute("COPY cityflow_attribute_predictions FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "cityflow_attribute_predictions.csv")))
