import json
from numpy import put
import requests.auth
import requests

def validate_snapshot(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    edges = data['edges']

    # Extract reference feature sets (ensures all nodes/edges have the same attributes)
    first_node = list(nodes.values())[0]
    first_edge = list(edges.values())[0]

    node_keys = set(first_node.keys())
    edge_keys = set(first_edge.keys())

    errors = []

    # Function to check if a feature correctly has a `has_*` flag
    def check_has_feature(entity_name, entity_data, feature_name, entity_type):
        has_key = f'has_{feature_name}'
        if feature_name in entity_data:
            value = entity_data[feature_name]
            has_value = entity_data.get(has_key, None)

            if feature.endswith("type"):
                return
            # If value is -1, has_feature should be 0
            if value == -1 and has_value != 0:
                errors.append(f"{entity_type} '{entity_name}' has {feature_name}=-1 but {has_key}={has_value} (should be 0)")

            # If value is valid, has_feature should be 1
            if value != -1 and has_value is not None and has_value != 1:
                errors.append(f"{entity_type} '{entity_name}' has valid {feature_name}={value} but {has_key}={has_value} (should be 1)")

            print(feature_name, value)
            if value != -1 and value < 0 or value > 1:
                errors.append(f"{entity_type} '{entity_name}' has invalid {feature_name}={value} (should be in [0, 1])")
    # Check all nodes
    for node_name, node_data in nodes.items():
        # Ensure all nodes have the same keys
        if set(node_data.keys()) != node_keys:
            errors.append(f"Node '{node_name}' has inconsistent keys: {set(node_data.keys())}")

        # Check all node attributes
        for feature in node_keys:
            if feature.startswith("has_"):  # Skip `has_*` itself
                continue
            check_has_feature(node_name, node_data, feature, "Node")

    # Check all edges
    for edge_name, edge_data in edges.items():
        # Ensure all edges have the same keys
        if set(edge_data.keys()) != edge_keys:
            errors.append(f"Edge '{edge_name}' has inconsistent keys: {set(edge_data.keys())}")

        # Check all edge attributes
        for feature in edge_keys:
            if feature.startswith("has_"):  # Skip `has_*` itself
                continue
            check_has_feature(edge_name, edge_data, feature, "Edge")

    # Print results
    if errors:
        print(f"❌ Found {len(errors)} issues:")
        for error in errors:
            print(f" - {error}")
    else:
        print("✅ Snapshot is correctly formatted and consistent!")

# Run the validation
#validate_snapshot("snapshot_2.json")

#def put_thing(device, namespace="wdn", ditto_url="localhost",port=8080):
#    epanet_point = device.get('epanet_point')
#    truth_point = device.get('truth_point')
#
#    device_id = truth_point['dev_eui']
#
#    url = f"http://{ditto_url}:{port}/api/2/things/{namespace}:{device_id}"
#    headers = {
#        "Content-Type": "application/json"
#    }
#    auth = requests.auth.HTTPBasicAuth('ditto', 'ditto')
#    data = {
#        "policyId": f"{namespace}:policy",
#        "attributes": {
#            "epanet_junction_name": epanet_point['id'],
#            "epanet_coords": {
#                "lat": epanet_point['lat'],
#                "lon": epanet_point['lon']
#            },
#            "device_name": truth_point['name'],
#            "dev_eui": truth_point['dev_eui'],
#            "real_coords": {
#                "lat": truth_point['lat'],
#                "lon": truth_point['lon']
#            },
#        },
#        "features": {}
#    }
#
#    response = requests.put(url, headers=headers, auth=auth, json=data)
#
#    if response.status_code >= 200:
#        print(f"PUT request to /things was {response.status_code}")
#        print(response.text)
#    else:
#        print(f"PUT request to /things failed with status code {response.status_code}")
#        print(response.text)
#
#
#with open('filtered_mapping.json', 'r') as f:
#    data = json.load(f)
#    for device in data:
#        print(device)
#        if device['truth_point']['dev_eui'] == "8C83FC05005B036A":
#            put_thing(device)

import json
import networkx as nx
import matplotlib.pyplot as plt

# Load JSON file
with open("data.json", "r") as f:
    data = json.load(f)

# Create an undirected graph
G = nx.Graph()


i = 0
# Add nodes and edges
for person, friends in data.items():
    i += 1
    if i > 100:
        break
    for friend in friends:
        G.add_edge(person, friend)

# Plotting (Warning: large graphs may be very slow or unreadable!)
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # Use spring layout for better spacing
nx.draw(G, pos, node_size=10, edge_color='gray', with_labels=False)
plt.title("Friendship Network")
plt.show()