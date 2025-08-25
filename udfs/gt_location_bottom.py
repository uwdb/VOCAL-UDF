def gt_0(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width):
    cy1 = (o1_y1 + o1_y2) / 2
    return cy1 >= 160 and cy1 <= 320

def gt_1(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width):
    cy1 = (o1_y1 + o1_y2) / 2
    return cy1 > 220 and cy1 <= 320

def gt_2(o1_oname, o1_x1, o1_y1, o1_x2, o1_y2, o1_anames, height, width):
    return o1_y1 >= 160 and o1_y1 <= 320