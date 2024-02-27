def gt_0(o1, o2):
    # Check if the right edge of o1 is to the left of the left edge of o2 by at least 20 pixels
    return o1['x2'] < o2['x1'] - 20

def gt_1(o1):
    return o1['x1'] >= 0 and o1['x2'] <= 240

def gt_2(o1):
    return o1['x1'] >= 0 and o1['x2'] <= 120

def gt_3(o1):
    cx = (o1['x1'] + o1['x2']) / 2
    return cx >= 0 and cx <= 120

def gt_4(o1):
    cx = (o1['x1'] + o1['x2']) / 2
    return cx >= 0 and cx <= 200
