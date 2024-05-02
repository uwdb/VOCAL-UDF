import duckdb
import pandas as pd
import time
from typing import List
import resource

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

#### Old UDFs ####
def left_of(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    return cx1 < cx2

def right_of(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    return cx1 > cx2

def front_of(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    cy1 = (o1_y1 + o1_y2) / 2
    cy2 = (o2_y1 + o2_y2) / 2
    return cy1 > cy2

def behind(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    cy1 = (o1_y1 + o1_y2) / 2
    cy2 = (o2_y1 + o2_y2) / 2
    return cy1 < cy2

def equal_size(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    area1 = (o1_x2-o1_x1)*(o1_y2-o1_y1)
    area2 = (o2_x2-o2_x1)*(o2_y2-o2_y1)
    return (area1 > 2400 and area2 > 2400) or (area1 <= 2400 and area2 <= 2400)

def equal_material(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    return o1_material == o2_material

def equal_shape(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    return o1_shape == o2_shape

def equal_color(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    return o1_color == o2_color

def color(color_name: str, o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == color_name

def shape(shape_name: str, o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_shape == shape_name

def material(material_name: str, o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_material == material_name

def size(size_name: str, o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    area = (o1_x2-o1_x1)*(o1_y2-o1_y1)
    if size_name == "small":
        return area <= 2400
    elif size_name == "big":
        return area > 2400
    else:
        raise ValueError("Unknown size_name: {}".format(size_name))

def size_big(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    area = (o1_x2-o1_x1)*(o1_y2-o1_y1)
    return area > 2400

def size_small(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    area = (o1_x2-o1_x1)*(o1_y2-o1_y1)
    return area <= 2400

def color_gray(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == "gray"

def color_red(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == "red"

def color_blue(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == "blue"

def color_green(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == "green"

def color_brown(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == "brown"

def color_purple(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == "purple"

def color_cyan(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == "cyan"

def color_yellow(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_color == "yellow"

def shape_cube(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_shape == "cube"

def shape_sphere(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_shape == "sphere"

def shape_cylinder(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_shape == "cylinder"

def material_rubber(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_material == "rubber"

def material_metal(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    return o1_material == "metal"

def far(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    import math
    cx1 = (o1_x1 + o1_x2) / 2
    cy1 = (o1_y1 + o1_y2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    cy2 = (o2_y1 + o2_y2) / 2

    distance = math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0)) / ((o1_x2 - o1_x1 + o2_x2 - o2_x1) / 2)
    return distance > 3.0

def near(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float, o2_shape: str, o2_color: str, o2_material: str, o2_x1: float, o2_y1: float, o2_x2: float, o2_y2: float) -> bool:
    import math
    cx1 = (o1_x1 + o1_x2) / 2
    cy1 = (o1_y1 + o1_y2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    cy2 = (o2_y1 + o2_y2) / 2

    distance = math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0)) / ((o1_x2 - o1_x1 + o2_x2 - o2_x1) / 2)
    return distance < 1.0

def left(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    return cx1 >= 0 and cx1 < 240

def right(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    return cx1 >= 240 and cx1 <= 480

def top(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    cy1 = (o1_y1 + o1_y2) / 2
    return cy1 >= 0 and cy1 < 160

def bottom(o1_shape: str, o1_color: str, o1_material: str, o1_x1: float, o1_y1: float, o1_x2: float, o1_y2: float) -> bool:
    cy1 = (o1_y1 + o1_y2) / 2
    return cy1 >= 160 and cy1 <= 320

##### New UDFs #####
def new_near(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], o2_oname: str, o2_x1: int, o2_y1: int, o2_x2: int, o2_y2: int, o2_anames: List[str], o1_o2_rnames: List[str], o2_o1_rnames: List[str], height: int, width: int) -> bool:
    import math
    # Calculate the center points of both objects
    cx1 = (o1_x1 + o1_x2) / 2
    cy1 = (o1_y1 + o1_y2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    cy2 = (o2_y1 + o2_y2) / 2

    # Calculate the distance between the center points, normalized by the average width of the objects
    distance = math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0)) / ((o1_x2 - o1_x1 + o2_x2 - o2_x1) / 2)

    threshold = 1.0

    return distance < threshold

def new_rightof(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], o2_oname: str, o2_x1: int, o2_y1: int, o2_x2: int, o2_y2: int, o2_anames: List[str], o1_o2_rnames: List[str], o2_o1_rnames: List[str], height: int, width: int) -> bool:
    # Check if the center of o1 is to the right of the center of o2
    cx_o1 = (o1_x1 + o1_x2) / 2
    cx_o2 = (o2_x1 + o2_x2) / 2
    return cx_o1 > cx_o2

def new_behind(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], o2_oname: str, o2_x1: int, o2_y1: int, o2_x2: int, o2_y2: int, o2_anames: List[str], o1_o2_rnames: List[str], o2_o1_rnames: List[str], height: int, width: int) -> bool:
    center_y_o1 = (o1_y1 + o1_y2) / 2
    center_y_o2 = (o2_y1 + o2_y2) / 2
    return center_y_o1 < center_y_o2

def new_frontof(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], o2_oname: str, o2_x1: int, o2_y1: int, o2_x2: int, o2_y2: int, o2_anames: List[str], o1_o2_rnames: List[str], o2_o1_rnames: List[str], height: int, width: int) -> bool:
    center_y_o1 = (o1_y1 + o1_y2) / 2
    center_y_o2 = (o2_y1 + o2_y2) / 2
    return center_y_o1 > center_y_o2

def new_color_brown(o1_oname: str, o1_x1: int, o1_y1: int, o1_x2: int, o1_y2: int, o1_anames: List[str], height: int, width: int) -> bool:
    return 'color_brown' in o1_anames

################################
################################
### test available UDFs only ###
################################
################################

def test1(conn):
    # color_red(o0), color_blue(o1), color_green(o2)
    print("Running test1...")
    print("Computing df1...")
    _start = time.time()
    df1 = conn.execute(f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
        FROM obj_attr_filtered as o1, obj_attr_filtered as o0, obj_attr_filtered as o2
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid
            and 'color_red' = ANY(o0.o1_anames) and 'color_blue' = ANY(o1.o1_anames) and 'color_green' = ANY(o2.o1_anames)
    """).df()
    print("df1 time:", time.time() - _start) # 94s

    print("Computing df2...")
    _start = time.time()
    df2 = conn.execute(f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid
        FROM Obj_filtered as o1, Obj_filtered as o0, Obj_filtered as o2
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and old_Color_red(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and old_Color_blue(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and old_Color_green(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;
    """).df()
    print("df2 time:", time.time() - _start) # 15s

    print("df1 length:", len(df1))
    print("df2 length:", len(df2))
    print(pd.concat([df1,df2]).drop_duplicates(keep=False).to_string())

def test2(conn):
    # Near(o0, o1), RightOf(o1, o2)
    print("Running test2...")
    print("Computing df1...")
    _start = time.time()
    df1 = conn.execute(f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
        FROM obj_attr_filtered as o1, obj_attr_filtered as o0, obj_attr_filtered as o2,
        rel_filtered as r1,
        rel_filtered as r2
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid
            and o0.vid = r1.vid and o0.fid = r1.fid and o0.o1_oid = r1.o1_oid and o1.o1_oid = r1.o2_oid and 'near' = ANY(r1.o1_o2_rnames) -- available relationship UDF
            and o1.vid = r2.vid and o1.fid = r2.fid and o1.o1_oid = r2.o1_oid and o2.o1_oid = r2.o2_oid and 'right_of' = ANY(r2.o1_o2_rnames) -- available relationship UDF
    """).df()
    print("df1 time:", time.time() - _start) # 76s

    print("Computing df2...")
    _start = time.time()
    df2 = conn.execute(f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid
        FROM Obj_filtered as o1, Obj_filtered as o0, Obj_filtered as o2
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid
            and old_Near(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true
            and old_RightOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true
            and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;
    """).df()
    print("df2 time:", time.time() - _start) # 145s

    print("df1 length:", len(df1))
    print("df2 length:", len(df2))
    print(pd.concat([df1,df2]).drop_duplicates(keep=False).to_string())


def test3(conn):
    # Near(o0, o1), RightOf(o1, o2), shape_cube(o0), shape_cube(o1)
    print("Running test3...")
    print("Computing df1...")
    _start = time.time()
    sql1 = f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
        FROM obj_attr_filtered as o1, obj_attr_filtered as o0, obj_attr_filtered as o2,
        rel_filtered as r1,
        rel_filtered as r2
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid
            and o0.vid = r1.vid and o0.fid = r1.fid and o0.o1_oid = r1.o1_oid and o1.o1_oid = r1.o2_oid and 'near' = ANY(r1.o1_o2_rnames)
            and o1.vid = r2.vid and o1.fid = r2.fid and o1.o1_oid = r2.o1_oid and o2.o1_oid = r2.o2_oid and 'right_of' = ANY(r2.o1_o2_rnames)
            and 'shape_cube' = ANY(o0.o1_anames) and 'shape_cube' = ANY(o1.o1_anames)
    """
    df1 = conn.execute(sql1).df()
    print("df1 time:", time.time() - _start) # 80s
    # plan1 = conn.execute(f"EXPLAIN {sql1}").fetchall()
    # print(plan1)

    print("Computing df2...")
    _start = time.time()
    sql2 = f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid
        FROM Obj_filtered as o1, Obj_filtered as o0, Obj_filtered as o2
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid
            and old_Near(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true
            and old_RightOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true
            and old_Shape_cube(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true
            and old_Shape_cube(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true
            and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;
    """
    df2 = conn.execute(sql2).df()
    print("df2 time:", time.time() - _start) # 15s
    # plan2 = conn.execute(f"EXPLAIN {sql2}").fetchall()
    # print(plan2)

    print("df1 length:", len(df1))
    print("df2 length:", len(df2))
    print(pd.concat([df1,df2]).drop_duplicates(keep=False).to_string())


def test4(conn):
    # Near(o0, o1), RightOf(o1, o2), ColorBrown(o0), Behind(o2, o0), FrontOf(o1, o0)
    print("Running test4...")
    print("Computing df1...")
    _start = time.time()
    sql1 = f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
        FROM obj_attr_filtered as o1, obj_attr_filtered as o0, obj_attr_filtered as o2,
        rel_filtered as r1,
        rel_filtered as r2,
        rel_filtered as r3,
        rel_filtered as r4
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid
            and o0.vid = r1.vid and o0.fid = r1.fid and o0.o1_oid = r1.o1_oid and o1.o1_oid = r1.o2_oid and 'near' = ANY(r1.o1_o2_rnames)
            and o1.vid = r2.vid and o1.fid = r2.fid and o1.o1_oid = r2.o1_oid and o2.o1_oid = r2.o2_oid and 'right_of' = ANY(r2.o1_o2_rnames)
            and 'color_brown' = ANY(o0.o1_anames)
            and o2.vid = r3.vid and o2.fid = r3.fid and o2.o1_oid = r3.o1_oid and o0.o1_oid = r3.o2_oid and 'behind' = ANY(r3.o1_o2_rnames)
            and o1.vid = r4.vid and o1.fid = r4.fid and o1.o1_oid = r4.o1_oid and o0.o1_oid = r4.o2_oid and 'front_of' = ANY(r4.o1_o2_rnames)
    """
    df1 = conn.execute(sql1).df()
    print("df1 time:", time.time() - _start) # 115s
    # plan1 = conn.execute(f"EXPLAIN {sql1}").fetchall()
    # print(plan1)

    print("Computing df2...")
    _start = time.time()
    sql2 = f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid
        FROM Obj_filtered as o1, Obj_filtered as o0, Obj_filtered as o2
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid
            and old_Near(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true
            and old_RightOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true
            and old_Color_brown(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true
            and old_Behind(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true
            and old_FrontOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true
            and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;
    """
    df2 = conn.execute(sql2).df()
    print("df2 time:", time.time() - _start) # 20s
    # plan2 = conn.execute(f"EXPLAIN {sql2}").fetchall()
    # print(plan2)

    print("df1 length:", len(df1))
    print("df2 length:", len(df2))
    print(pd.concat([df1,df2]).drop_duplicates(keep=False).to_string())

#######################
#######################
### test on the fly ###
#######################
#######################

def test5(conn):
    # Near(o0, o1), RightOf(o1, o2), ColorBrown(o0), Behind(o2, o0), FrontOf(o1, o0)
    print("Running test5...")
    print("Computing df1...")
    _start = time.time()
    sql1 = f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
        FROM obj_attr_filtered as o1, obj_attr_filtered as o0, obj_attr_filtered as o2,
        rel_filtered as r1,
        rel_filtered as r2,
        rel_filtered as r3,
        rel_filtered as r4
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid
            and o0.vid = r1.vid and o0.fid = r1.fid and o0.o1_oid = r1.o1_oid and o1.o1_oid = r1.o2_oid and new_Near(r1.o1_oname, r1.o1_x1, r1.o1_y1, r1.o1_x2, r1.o1_y2, r1.o1_anames, r1.o2_oname, r1.o2_x1, r1.o2_y1, r1.o2_x2, r1.o2_y2, r1.o2_anames, r1.o1_o2_rnames, r1.o2_o1_rnames, r1.height, r1.width) = true
            and o1.vid = r2.vid and o1.fid = r2.fid and o1.o1_oid = r2.o1_oid and o2.o1_oid = r2.o2_oid and new_RightOf(r2.o1_oname, r2.o1_x1, r2.o1_y1, r2.o1_x2, r2.o1_y2, r2.o1_anames, r2.o2_oname, r2.o2_x1, r2.o2_y1, r2.o2_x2, r2.o2_y2, r2.o2_anames, r2.o1_o2_rnames, r2.o2_o1_rnames, r2.height, r2.width) = true
            and new_Color_brown(o0.o1_oname, o0.o1_x1, o0.o1_y1, o0.o1_x2, o0.o1_y2, o0.o1_anames, o0.height, o0.width) = true
            and o2.vid = r3.vid and o2.fid = r3.fid and o2.o1_oid = r3.o1_oid and o0.o1_oid = r3.o2_oid and new_Behind(r3.o1_oname, r3.o1_x1, r3.o1_y1, r3.o1_x2, r3.o1_y2, r3.o1_anames, r3.o2_oname, r3.o2_x1, r3.o2_y1, r3.o2_x2, r3.o2_y2, r3.o2_anames, r3.o1_o2_rnames, r3.o2_o1_rnames, r3.height, r3.width) = true
            and o1.vid = r4.vid and o1.fid = r4.fid and o1.o1_oid = r4.o1_oid and o0.o1_oid = r4.o2_oid and new_FrontOf(r4.o1_oname, r4.o1_x1, r4.o1_y1, r4.o1_x2, r4.o1_y2, r4.o1_anames, r4.o2_oname, r4.o2_x1, r4.o2_y1, r4.o2_x2, r4.o2_y2, r4.o2_anames, r4.o1_o2_rnames, r4.o2_o1_rnames, r4.height, r4.width) = true
    """
    df1 = conn.execute(sql1).df()
    print("df1 time:", time.time() - _start) # 537s
    print(using('profile'))

    print("Computing df2...")
    _start = time.time()
    sql2 = f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
        FROM obj_attr_filtered as o1, obj_attr_filtered as o0, obj_attr_filtered as o2,
        rel_filtered as r1,
        rel_filtered as r2,
        rel_filtered as r3,
        rel_filtered as r4
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid
            and o0.vid = r1.vid and o0.fid = r1.fid and o0.o1_oid = r1.o1_oid and o1.o1_oid = r1.o2_oid and 'near' = ANY(r1.o1_o2_rnames)
            and o1.vid = r2.vid and o1.fid = r2.fid and o1.o1_oid = r2.o1_oid and o2.o1_oid = r2.o2_oid and 'right_of' = ANY(r2.o1_o2_rnames)
            and 'color_brown' = ANY(o0.o1_anames)
            and o2.vid = r3.vid and o2.fid = r3.fid and o2.o1_oid = r3.o1_oid and o0.o1_oid = r3.o2_oid and 'behind' = ANY(r3.o1_o2_rnames)
            and o1.vid = r4.vid and o1.fid = r4.fid and o1.o1_oid = r4.o1_oid and o0.o1_oid = r4.o2_oid and 'front_of' = ANY(r4.o1_o2_rnames)
    """
    df2 = conn.execute(sql2).df()
    print("df2 time:", time.time() - _start) # 100s
    print(using('profile'))

    print("df1 length:", len(df1))
    print("df2 length:", len(df2))
    print(pd.concat([df1,df2]).drop_duplicates(keep=False).to_string())


#########################
#########################
### test materialized ###
#########################
#########################

def test6(conn):
    # Near(o0, o1), RightOf(o1, o2), ColorBrown(o0), Behind(o2, o0), FrontOf(o1, o0)
    print("Running test6...")
    # Construct the materialized tables
    print("Computing df_near...")
    df_near = conn.execute("""
        SELECT vid, fid,
            o1_oid, o1_oid, o1_x1, o1_y1, o1_x2, o1_y2,
            o2_oid, o2_oid, o2_x1, o2_y1, o2_x2, o2_y2,
            ('near' = ANY(o1_o2_rnames)) as pred
        FROM rel_filtered
    """).df()
    # df_near['pred'] = df_near['o1_o2_rnames'].apply(lambda x: int('near' in x))
    print(using('profile'))

    print("Computing df_rightof...")
    df_rightof = conn.execute("""
        SELECT vid, fid,
            o1_oid, o1_oid, o1_x1, o1_y1, o1_x2, o1_y2,
            o2_oid, o2_oid, o2_x1, o2_y1, o2_x2, o2_y2,
            ('right_of' = ANY(o1_o2_rnames)) as pred
        FROM rel_filtered
    """).df()
    # df_rightof['pred'] = df_rightof['o1_o2_rnames'].apply(lambda x: int('right_of' in x))
    print(using('profile'))

    print("Computing df_color_brown...")
    df_color_brown = conn.execute("""
        SELECT vid, fid,
            o1_oid, o1_oid, o1_x1, o1_y1, o1_x2, o1_y2,
            ('color_brown' = ANY(o1_anames)) as pred
        FROM obj_attr_filtered
    """).df()
    # df_color_brown['pred'] = df_color_brown['o1_anames'].apply(lambda x: int('color_brown' in x))
    print(using('profile'))

    print("Computing df_behind...")
    df_behind = conn.execute("""
        SELECT vid, fid,
            o1_oid, o1_oid, o1_x1, o1_y1, o1_x2, o1_y2,
            o2_oid, o2_oid, o2_x1, o2_y1, o2_x2, o2_y2,
            ('behind' = ANY(o1_o2_rnames)) as pred
        FROM rel_filtered
    """).df()
    # df_behind['pred'] = df_behind['o1_o2_rnames'].apply(lambda x: int('behind' in x))
    print(using('profile'))

    print("Computing df_frontof...")
    df_frontof = conn.execute("""
        SELECT vid, fid,
            o1_oid, o1_oid, o1_x1, o1_y1, o1_x2, o1_y2,
            o2_oid, o2_oid, o2_x1, o2_y1, o2_x2, o2_y2,
            ('front_of' = ANY(o1_o2_rnames)) as pred
        FROM rel_filtered
    """).df()
    # df_frontof['pred'] = df_frontof['o1_o2_rnames'].apply(lambda x: int('front_of' in x))
    print(using('profile'))

    print("Computing df1...")
    _start = time.time()
    sql1 = f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
        FROM obj_attr_filtered as o1, obj_attr_filtered as o0, obj_attr_filtered as o2,
        df_near as r1,
        df_rightof as r2,
        df_behind as r3,
        df_frontof as r4,
        df_color_brown as a1
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid
            and o0.vid = r1.vid and o0.fid = r1.fid and o0.o1_oid = r1.o1_oid and o1.o1_oid = r1.o2_oid and r1.pred = 1
            and o1.vid = r2.vid and o1.fid = r2.fid and o1.o1_oid = r2.o1_oid and o2.o1_oid = r2.o2_oid and r2.pred = 1
            and a1.vid = o0.vid and a1.fid = o0.fid and a1.o1_oid = o0.o1_oid and a1.pred = 1
            and o2.vid = r3.vid and o2.fid = r3.fid and o2.o1_oid = r3.o1_oid and o0.o1_oid = r3.o2_oid and r3.pred = 1
            and o1.vid = r4.vid and o1.fid = r4.fid and o1.o1_oid = r4.o1_oid and o0.o1_oid = r4.o2_oid and r4.pred = 1
    """
    df1 = conn.execute(sql1).df()
    print("df1 time:", time.time() - _start) # 537s
    print(using('profile'))

    print("Computing df2...")
    _start = time.time()
    sql2 = f"""
        SELECT DISTINCT o0.vid as vid, o0.fid as fid, o0.o1_oid as o0_oid, o1.o1_oid as o1_oid, o2.o1_oid as o2_oid
        FROM obj_attr_filtered as o1, obj_attr_filtered as o0, obj_attr_filtered as o2,
        rel_filtered as r1,
        rel_filtered as r2,
        rel_filtered as r3,
        rel_filtered as r4
        WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and o0.o1_oid <> o1.o1_oid and o0.o1_oid <> o2.o1_oid and o1.o1_oid <> o2.o1_oid
            and o0.vid = r1.vid and o0.fid = r1.fid and o0.o1_oid = r1.o1_oid and o1.o1_oid = r1.o2_oid and 'near' = ANY(r1.o1_o2_rnames)
            and o1.vid = r2.vid and o1.fid = r2.fid and o1.o1_oid = r2.o1_oid and o2.o1_oid = r2.o2_oid and 'right_of' = ANY(r2.o1_o2_rnames)
            and 'color_brown' = ANY(o0.o1_anames)
            and o2.vid = r3.vid and o2.fid = r3.fid and o2.o1_oid = r3.o1_oid and o0.o1_oid = r3.o2_oid and 'behind' = ANY(r3.o1_o2_rnames)
            and o1.vid = r4.vid and o1.fid = r4.fid and o1.o1_oid = r4.o1_oid and o0.o1_oid = r4.o2_oid and 'front_of' = ANY(r4.o1_o2_rnames)
    """
    df2 = conn.execute(sql2).df()
    print("df2 time:", time.time() - _start) # 100s
    print(using('profile'))

    print("df1 length:", len(df1))
    print("df2 length:", len(df2))
    print(pd.concat([df1,df2]).drop_duplicates(keep=False).to_string())


if __name__ == "__main__":
    conn = duckdb.connect(database="/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb", read_only=True)
    # conn.execute("SET memory_limit='40GB';")
    # conn.execute("SET threads TO 2;")
    # conn.execute("SET temp_directory = '/gscratch/balazinska/enhaoz/VOCAL-UDF/duckdb_dir/annotations.duckdb.tmp/';")

    conn.create_function("old_LeftOf", left_of)
    conn.create_function("old_RightOf", right_of)
    conn.create_function("old_FrontOf", front_of)
    conn.create_function("old_Behind", behind)
    conn.create_function("old_EqualSize", equal_size)
    conn.create_function("old_EqualMaterial", equal_material)
    conn.create_function("old_EqualShape", equal_shape)
    conn.create_function("old_EqualColor", equal_color)
    conn.create_function("old_Color", color)
    conn.create_function("old_Shape", shape)
    conn.create_function("old_Material", material)
    conn.create_function("old_Size", size)

    # BIG, SMALL, GRAY, RED, BLUE, GREEN, BROWN, PURPLE, CYAN, YELLOW, CUBE, SPHERE, CYLINDER, RUBBER, METAL.
    conn.create_function("old_Size_big", size_big)
    conn.create_function("old_Size_small", size_small)
    conn.create_function("old_Color_gray", color_gray)
    conn.create_function("old_Color_red", color_red)
    conn.create_function("old_Color_blue", color_blue)
    conn.create_function("old_Color_green", color_green)
    conn.create_function("old_Color_brown", color_brown)
    conn.create_function("old_Color_purple", color_purple)
    conn.create_function("old_Color_cyan", color_cyan)
    conn.create_function("old_Color_yellow", color_yellow)
    conn.create_function("old_Shape_cube", shape_cube)
    conn.create_function("old_Shape_sphere", shape_sphere)
    conn.create_function("old_Shape_cylinder", shape_cylinder)
    conn.create_function("old_Material_rubber", material_rubber)
    conn.create_function("old_Material_metal", material_metal)

    # For CLEVRER
    conn.create_function("old_Far", far)
    conn.create_function("old_Near", near)
    conn.create_function("old_Left", left)
    conn.create_function("old_Right", right)
    conn.create_function("old_Top", top)
    conn.create_function("old_Bottom", bottom)

    # For new UDFs
    conn.create_function("new_Near", new_near)
    conn.create_function("new_RightOf", new_rightof)
    conn.create_function("new_Behind", new_behind)
    conn.create_function("new_FrontOf", new_frontof)
    conn.create_function("new_Color_brown", new_color_brown)

    dataset = "clevrer"
    input_vids = 10000
    attribute_domain = ['color_gray', 'color_red', 'color_blue', 'color_green', 'color_brown', 'color_purple', 'color_cyan', 'color_yellow', 'shape_cylinder', 'shape_cube', 'shape_sphere', 'material_metal', 'material_rubber', 'location_right', 'location_top', 'location_bottom', 'location_left']
    relationship_domain = ['near', 'far', 'left_of', 'right_of', 'front_of', 'behind']
    attr_parameters = ','.join('?' for _ in attribute_domain)
    rel_parameters = ','.join('?' for _ in relationship_domain)

    print("Creating Obj_filtered...")
    conn.execute(f"CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM Obj_clevrer WHERE vid < {input_vids};")

    print("Creating one_object...")
    conn.execute(f"""
        CREATE TEMPORARY TABLE one_object ON COMMIT DROP AS
        SELECT
            o.vid AS vid, o.fid AS fid, o.oid AS o1_oid, o.oname AS o1_oname,
            o.x1 AS o1_x1, o.y1 AS o1_y1, o.x2 AS o1_x2, o.y2 AS o1_y2,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname IS NOT NULL), ARRAY[]::varchar[]) AS o1_gt_anames,
            COALESCE(ARRAY_AGG(a.aname) FILTER (WHERE a.aname = ANY([{attr_parameters}])), ARRAY[]::varchar[]) AS o1_anames,
            320 AS height, 480 AS width
        FROM clevrer_objects o
        LEFT OUTER JOIN clevrer_attributes a ON o.vid = a.vid AND o.fid = a.fid AND o.oid = a.oid
        WHERE o.vid < {input_vids}
        GROUP BY o.vid, o.fid, o.oid, o.oname, o.x1, o.y1, o.x2, o.y2;
    """, attribute_domain)

    print("Creating two_objects...")
    conn.execute(f"""
        CREATE TEMPORARY TABLE two_objects ON COMMIT DROP AS
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
        WHERE o1.oid <> o2.oid and o1.vid < {input_vids}
    """, attribute_domain + relationship_domain)

    conn.execute(f"CREATE TEMPORARY TABLE obj_attr_filtered AS SELECT * FROM one_object WHERE vid < {input_vids};")
    conn.execute(f"CREATE TEMPORARY TABLE rel_filtered AS SELECT * FROM two_objects WHERE vid < {input_vids};")
    print(using('profile'))
    # test1(conn)
    # test2(conn)
    # test3(conn)
    # test4(conn)
    # test5(conn)
    test6(conn)

    print("peak memory usage (kb): ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)