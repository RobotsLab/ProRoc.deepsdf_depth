import os
import json
import numpy as np

# Define a function to load all json files and compute min/max values
def find_min_max_values(path):
    min_u = float('inf')
    max_u = float('-inf')
    min_v = float('inf')
    max_v = float('-inf')
    
    min_dd = float('inf')
    max_dd = float('-inf')
    
    min_rd = float('inf')
    max_rd = float('-inf')
    
    min_sdf = float('inf')
    max_sdf = float('-inf')

    for category in ['bottle', 'mug', 'bowl']:
        directory = os.path.join(path, category)
        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            if filename.endswith("k200_inp_train.json"):
                filepath = os.path.join(directory, filename)
                print(f"Processing file: {filename}")

                # Load the JSON file
                with open(filepath, 'r') as f:
                    input_file = json.load(f)

                # Iterate over the keys (which represent the u, v pairs) and values
                for key, value in input_file.items():
                    u = int(key.split(', ')[0])
                    v = int(key.split(', ')[1])

                    # Update min/max values for u and v
                    min_u = min(min_u, u)
                    max_u = max(max_u, u)
                    min_v = min(min_v, v)
                    max_v = max(max_v, v)

                    # Iterate over each point in the value list
                    for point in value:
                        try:
                            dd = point[0]
                            rd = point[1]
                            sdf = point[2]
                        except:
                            print(point, type(point[0]))

                        # Update min/max for dd, rd, and sdf
                        min_dd = min(min_dd, dd)
                        max_dd = max(max_dd, dd)

                        min_rd = min(min_rd, rd)
                        max_rd = max(max_rd, rd)

                        min_sdf = min(min_sdf, sdf)
                        max_sdf = max(max_sdf, sdf)

    return {
        'min_u': min_u, 'max_u': max_u,
        'min_v': min_v, 'max_v': max_v,
        'min_dd': min_dd, 'max_dd': max_dd,
        'min_rd': min_rd, 'max_rd': max_rd,
        'min_sdf': min_sdf, 'max_sdf': max_sdf
    }

# c:\Users\micha\OneDrive\Pulpit\DeepSDF\examples\new_exp_10\data\training_data\bottle\10f709cecfbb8d59c2536abb1e8e5eab_5_a25_view0_k200_inp_train.json
# Example usage: specify the directory containing your JSON files
path = 'examples/new_exp_10/data/training_data'  # Replace with the correct path
min_max_values = find_min_max_values(path)

# Print the min/max values found across all files
print("Min and Max Values:")
for key, value in min_max_values.items():
    print(f"{key}: {value}")


# min_u: 534
# max_u: 711
# min_v: 232
# max_v: 419
# min_dd: 0.10000000000000009
# max_dd: 0.30145583152771005
# min_rd: -0.30101666450500497
# max_rd: 0.3999999999999999
# min_sdf: 0
# max_sdf: 0.2930116653442383