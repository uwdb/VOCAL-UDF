def gt_0(o1, o2):
    import math
    # Calculate the center points of both objects
    cx1 = (o1['x1'] + o1['x2']) / 2
    cy1 = (o1['y1'] + o1['y2']) / 2
    cx2 = (o2['x1'] + o2['x2']) / 2
    cy2 = (o2['y1'] + o2['y2']) / 2

    # Calculate the distance between the center points, normalized by the average width of the objects
    distance = math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0)) / ((o1['x2'] - o1['x1'] + o2['x2'] - o2['x1']) / 2)

    threshold = 1.0

    return distance < threshold