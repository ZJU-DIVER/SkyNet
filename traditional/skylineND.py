def is_dominated_by(record1, record2):
    assert len(record1) == len(record2)
    for i in range(len(record2)):
        if record2[i] > record1[i]:
            return False

    if sum(record1) == sum(record2):
        return False

    return True


def skyline(data):
    skyline_points = []
    skyline_index = []
    for i in range(len(data)):
        cur = data[i]
        is_skyline = True
        for point in skyline_points:
            if is_dominated_by(cur, point):
                is_skyline = False
                break

        if not is_skyline:
            continue

        for j in range(i + 1, len(data)):
            if is_dominated_by(cur, data[j]):
                is_skyline = False
                break

        if is_skyline:
            skyline_points.append(cur)
            skyline_index.append(i + 1)

    return skyline_index, skyline_points
