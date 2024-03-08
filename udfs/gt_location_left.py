def gt_0(o1):
    cx = (o1['x1'] + o1['x2']) / 2
    return cx >= 0 and cx < 240

def gt_1(o1):
    return o1['x1'] >= 0 and o1['x2'] <= 240

def gt_2(o1):
    return o1['x1'] >= 0 and o1['x2'] <= 120

def gt_3(o1):
    cx = (o1['x1'] + o1['x2']) / 2
    return cx >= 0 and cx <= 200
