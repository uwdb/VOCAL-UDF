def gt_0(o1, o2):
    center_y_o1 = (o1['y1'] + o1['y2']) / 2
    center_y_o2 = (o2['y1'] + o2['y2']) / 2
    return center_y_o1 < center_y_o2

def gt_1(o1, o2):
    return o1['y1'] < o2['y1']

def gt_2(o1, o2):
    return o1['y2'] < o2['y2']

def gt_3(o1, o2):
    return o1['y2'] < o2['y1']

def gt_4(o1, o2):
    center_y_o1 = (o1['y1'] + o1['y2']) / 2
    center_y_o2 = (o2['y1'] + o2['y2']) / 2
    return center_y_o1 < center_y_o2 - 20