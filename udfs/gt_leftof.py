def gt_0(o1, o2):
    # Check if the center of o1 is to the left of the center of o2
    cx_o1 = (o1['x1'] + o1['x2']) / 2
    cx_o2 = (o2['x1'] + o2['x2']) / 2
    return cx_o1 < cx_o2

def gt_1(o1, o2):
    # Check if the right edge of o1 is to the left of the left edge of o2
    return o1['x2'] < o2['x1']

def gt_2(o1, o2):
    # Check if the left edge of o1 is to the left of the left edge of o2
    return o1['x1'] < o2['x1']

def gt_3(o1, o2):
    # Check if the right edge of o1 is to the left of the right edge of o2
    return o1['x2'] < o2['x2']

def gt_4(o1, o2):
    # Check if the right edge of o1 is to the left of the left edge of o2 by at least 20 pixels
    return o1['x2'] < o2['x1'] - 20
