def gt_0(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    # Check if the center of o1 is to the left of the center of o2
    cx_o1 = (o1_x1 + o1_x2) / 2
    cx_o2 = (o2_x1 + o2_x2) / 2
    return cx_o1 < cx_o2

def gt_1(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    # Check if the right edge of o1 is to the left of the left edge of o2
    return o1_x2 < o2_x1

def gt_2(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    # Check if the left edge of o1 is to the left of the left edge of o2
    return o1_x1 < o2_x1

def gt_3(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    # Check if the right edge of o1 is to the left of the right edge of o2
    return o1_x2 < o2_x2

def gt_4(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, o2_oname, o2_x1, o2_y1, o2_x2, o2_y2, o2_anames, o1_o2_rnames, o2_o1_rnames, height, width):
    # Check if the right edge of o1 is to the left of the left edge of o2 by at least 20 pixels
    return o1_x2 < o2_x1 - 20
