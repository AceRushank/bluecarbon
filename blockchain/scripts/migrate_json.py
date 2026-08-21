import json
import os

paths = [
    "c:/Users/aceru/bluecarbon/blockchain/deployed_contracts.json",
    "c:/Users/aceru/bluecarbon/contracts_config.json"
]

for p in paths:
    if os.path.exists(p):
        with open(p, "r") as f:
            data = json.load(f)
        
        # If it's already multi-network (has "localhost", "amoy", etc as top level keys), skip
        if "network" in data:
            net_name = data["network"]
            new_data = {net_name: data}
            with open(p, "w") as f:
                json.dump(new_data, f, indent=2)
            print(f"Migrated {p} to multi-network structure.")
