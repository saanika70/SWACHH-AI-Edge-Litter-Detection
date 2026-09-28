from swachh_ai.geometry import in_zones, point_to_box_distance, proximity_ratio


def test_point_inside_box_is_zero():
    assert point_to_box_distance(5, 5, (0, 0, 10, 10)) == 0


def test_point_outside_box():
    assert point_to_box_distance(13, 14, (0, 0, 10, 10)) == 5.0  # 3-4-5 triangle


def test_proximity_is_normalised_by_person_height():
    person = (0, 0, 100, 200)
    litter = (150, 90, 170, 110)  # centre x=160 -> 60 px right of the box
    assert abs(proximity_ratio(litter, person) - 60 / 200) < 1e-9


def test_zone_membership():
    assert in_zones((500, 450), [[0.7, 0.8, 1.0, 1.0]], 640, 480)
    assert not in_zones((100, 100), [[0.7, 0.8, 1.0, 1.0]], 640, 480)
