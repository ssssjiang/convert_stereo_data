import rerun as rr

rr.init("rerun_example_log_file", spawn=True)

rr.log_file_from_path("/home/roborock/test_viz.rrd")
