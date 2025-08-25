def gt_0(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    center_y_o1 = (o1_y1 + o1_y2) / 2
    center_y_o2 = (o2_y1 + o2_y2) / 2
    return center_y_o1 < center_y_o2

def gt_1(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    center_y_o1 = (o1_y1 + o1_y2) / 2
    center_y_o2 = (o2_y1 + o2_y2) / 2
    return center_y_o1 > center_y_o2

def gt_2(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    return o1_y2 < o2_y2

def gt_3(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    return o1_y2 < o2_y1 - 20

def gt_4(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    center_y_o1 = (o1_y1 + o1_y2) / 2
    center_y_o2 = (o2_y1 + o2_y2) / 2
    return center_y_o1 < center_y_o2 - 20