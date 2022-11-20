def autorewards(dist, new_dist, newPosition):
    H = 0
    if dist <= 0.25:
        if new_dist <= dist:
            H += 10
        else:
            H += -10
    else:
        if new_dist < dist:
            H += 10
        else:
            H += -10
    if newPosition < 0:
        H = -10
    return H
