def gt_0(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    import math
    # Calculate the center points of both objects
    cx1 = (o1_x1 + o1_x2) / 2
    cy1 = (o1_y1 + o1_y2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    cy2 = (o2_y1 + o2_y2) / 2

    # Calculate the distance between the center points, normalized by the average width of the objects
    distance = math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0)) / ((o1_x2 - o1_x1 + o2_x2 - o2_x1) / 2)

    # Define a threshold for being "far"
    threshold = 3.0

    # Return whether the objects are far away each other
    return distance >= threshold

def gt_1(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    import math
    cx1 = (o1_x1 + o1_x2) / 2
    cy1 = (o1_y1 + o1_y2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    cy2 = (o2_y1 + o2_y2) / 2

    distance = math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0))

    threshold = 100

    return distance >= threshold

def gt_2(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    # Determine whether o0 is far from o1 by checking if their bounding boxes overlap; if they do not, then they are considered far away.
    if o1_x1 > o2_x2 or o1_x2 < o2_x1 or o1_y1 > o2_y2 or o1_y2 < o2_y1:
        return True
    else:
        return False

def gt_3(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    cx1 = (o1_x1 + o1_x2) / 2
    cy1 = (o1_y1 + o1_y2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    cy2 = (o2_y1 + o2_y2) / 2

    distance = abs(cx1 - cx2) + abs(cy1 - cy2)

    threshold = 100

    return distance >= threshold

def gt_4(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    import math
    # Calculate the center points of both objects
    cx1 = (o1_x1 + o1_x2) / 2
    cy1 = (o1_y1 + o1_y2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    cy2 = (o2_y1 + o2_y2) / 2

    # Calculate the distance between the center points, normalized by the average width of the objects
    distance = math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0)) / ((o1_x2 - o1_x1 + o2_x2 - o2_x1) / 2)

    # Define a threshold for being "far"
    threshold = 2.0

    # Return whether the objects are far away each other
    return distance >= threshold