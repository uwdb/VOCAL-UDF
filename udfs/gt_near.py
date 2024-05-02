def gt_0(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
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