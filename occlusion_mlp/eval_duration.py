import os
from train import train
import json
import numpy as np

hours = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
num_reps = 5

if __name__ == "__main__":
    results = {h: [] for h in hours}
    for h in hours:
        with open(os.path.join("training_configs", "eval_duration.json"), "r") as f:
            config = json.load(f)
        config["FILENAME"] = os.path.join("/",
                                          "data",
                                          "real-world",
                                          "eval_duration", 
                                          f"train_{str(np.round(h, 2)).replace('.', '_')}h.h5")
        for rep in range(num_reps):
            _, balanced_acc_1m_test = train(config_default=config)
            results[h].append(balanced_acc_1m_test)

    os.makedirs("output", exist_ok=True)
    with open(ouput_path:=os.path.join("output", "eval_duration.json"), "w") as f:
        json.dump(results, f)
    print(f"Saved results to {ouput_path}")



    

