import math

def pyfar(o1_x1, o1_y1, o1_x2, o1_y2, o2_x1, o2_y1, o2_x2, o2_y2):
    # Check if bounding boxes intersect
    if (o1_x1 < o2_x2 and o1_x2 > o2_x1 and o1_y1 < o2_y2 and o1_y2 > o2_y1):
        return False  # Bounding boxes intersect

    # Calculate the center points of both objects
    cx1 = (o1_x1 + o1_x2) / 2
    cy1 = (o1_y1 + o1_y2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    cy2 = (o2_y1 + o2_y2) / 2

    # Calculate the distance between the center points
    distance = math.sqrt(math.pow(cx1 - cx2, 2.0) + math.pow(cy1 - cy2, 2.0))

    # Define a threshold for being "far" based on the average dimensions of the objects
    average_dimension = ((o1_x2 - o1_x1 + o1_y2 - o1_y1 + o2_x2 - o2_x1 + o2_y2 - o2_y1) / 4)
    threshold = average_dimension * 1.5  # Adjust this multiplier as needed

    # Return whether the objects are far from each other
    return distance > threshold

if __name__ == '__main__':
    print(pyfar(378, 110, 426, 164, 64, 180, 125, 252))
    print(pyfar(60, 109, 100, 148, 271, 129, 317, 189))
    print(pyfar(134, 1, 158, 25, 363, 95, 423, 150))
    print(pyfar(180, 33, 209, 68, 332, 151, 386, 216))
    print(pyfar(1, 130, 22, 171, 294, 75, 331, 110))
    print(pyfar(339, 142, 406, 208, 120, 114, 161, 154))
    print(pyfar(201, 52, 232, 83, 109, 67, 157, 112))
    print(pyfar(24, 210, 84, 267, 57, 175, 109, 226))
    print(pyfar(193, 130, 244, 192, 15, 61, 56, 101))
    print(pyfar(251, 38, 285, 76, 296, 39, 327, 77))
