def test_stage_x_routes_through_microscope(stage_proxy):
    stage_proxy.x = 0
    assert stage_proxy.x == 67.0