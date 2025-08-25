import duckdb
import os
import yaml

project_root = os.getenv("PROJECT_ROOT")

config = yaml.safe_load(
    open(os.path.join(project_root, "configs", "config.yaml"), "r")
)
db_dir = config["db_dir"]

conn = duckdb.connect(database=os.path.join(db_dir, "annotations.duckdb"), read_only=False)

### Load Charades data ###
conn.execute("CREATE TABLE charades_metadata (vname varchar, vid INT, fid INT, width INT, height INT, split varchar);")
conn.execute("CREATE TABLE charades_objects (vid INT, fid INT, oid INT, oname varchar, x1 int, y1 int, x2 int, y2 int);")
conn.execute("CREATE TABLE charades_relationships (vid INT, fid INT, rid INT, oid1 INT, rname varchar, oid2 INT);")
conn.execute("CREATE TABLE charades_attributes (vid INT, fid INT, oid INT, aname varchar);") # An empty table
conn.execute("CREATE TABLE charades_relationship_predictions (vid INT, fid INT, rid INT, oid1 INT, rname varchar, oid2 INT);")
conn.execute("CREATE TABLE charades_attribute_predictions (vid INT, fid INT, oid INT, aname varchar);") # An empty table

conn.execute("COPY charades_metadata FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "charades_metadata.csv")))
conn.execute("COPY charades_objects FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "charades_objects.csv")))
conn.execute("COPY charades_relationships FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "charades_relationships.csv")))
conn.execute("COPY charades_relationship_predictions FROM '{}' (FORMAT 'csv', delimiter ',', header);".format(os.path.join(db_dir, "charades_relationship_predictions.csv")))