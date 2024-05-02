def gt_0(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width):
    cx = (o1_x1 + o1_x2) / 2
    return cx >= 0 and cx < 240

def gt_1(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width):
    return o1_x1 >= 0 and o1_x2 <= 240

def gt_2(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width):
    return o1_x1 >= 0 and o1_x2 <= 120

def gt_3(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width):
    cx = (o1_x1 + o1_x2) / 2
    return cx >= 0 and cx <= 200
