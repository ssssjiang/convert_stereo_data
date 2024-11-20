
import pandas as pd# Load the new file
new_file_path = '/home/roborock/repos/vslam/build_stereo/tests/pose.txt'
new_data = pd.read_csv(new_file_path, delim_whitespace=True, header=None)

# Sorting the new data based on the first column (timestamps)
sorted_new_data = new_data.sort_values(by=[0])

# Save the sorted data
sorted_new_file_path = '/home/roborock/repos/vslam/build_stereo/tests/pose_3.txt'
sorted_new_data.to_csv(sorted_new_file_path, sep=' ', header=False, index=False)

