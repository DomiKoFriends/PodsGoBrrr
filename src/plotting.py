from os import makedirs, path
import numpy as np
import matplotlib.pyplot as plt
import json
from sys import argv

def create_ploting_entry(data, title, xlabel, ylabel, x, y, plot_min=False, plot_max=False, plot_mean=True, plot_std=False, raw_data=False):
    return {
        "title" : title,
        "xlabel" : xlabel,
        "ylabel" : ylabel,
        "min" : plot_min,
        "max" : plot_max,
        "mean" : plot_mean,
        "std" : plot_std,
        "raw_data" : raw_data,
        "x_data" : data[x],
        "y_data" : data[y]
    }

def save_ploting_data(model_name, data):
    assure_directory_exists(model_name)
    file_path = path.join(".", "models", model_name, "figures", "ploting_data.json")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def assure_directory_exists(model_name):
    directory_path = path.join(".", "models", model_name, "figures")

    if not path.exists(directory_path):
        print(f"Creating directory: {directory_path}")
        makedirs(directory_path)

def plot_and_save_metric(entry, model_name):
    directory_path = path.join(".", "models", model_name, "figures")
    
    assure_directory_exists(model_name)

    x_data = np.array(entry["x_data"])
    y_data = np.array(entry["y_data"])

    max_points = 200
    
    height = 10
    width = 10 * ((x_data.shape[0] + max_points - 1) / max_points)

    fig = plt.figure(figsize=(width, height), dpi=100)

    if entry["min"]:
        plt.plot(x_data.tolist(), y_data.min(axis=1).tolist(),'--o', label="min")

    if entry["max"]:
        plt.plot(x_data.tolist(), y_data.max(axis=1).tolist(), "--o", label="max")
        
    if entry["mean"]:
        plt.plot(x_data.tolist(), y_data.mean(axis=1).tolist(), "--o", label="mean")
    
    if entry["std"]:
        y_mean = y_data.mean(axis=1)
        y_std = y_data.std(axis=1)
        # plt.plot(x_data, y_mean - y_std, alpha=0.3)
        # plt.plot(x_data, y_mean + y_std, alpha=0.3)
        plt.fill_between(x_data, y_mean - y_std, y_mean + y_std, alpha=0.3)

    if entry["raw_data"]:
        plt.plot(x_data.tolist(), y_data.tolist(), "--o")
    
    plt.title(entry["title"])
    plt.xlabel(entry["xlabel"])
    plt.ylabel(entry["ylabel"])

    plt.legend()
    plt.tight_layout()
    plt.savefig(path.join(directory_path, entry["title"]))
    plt.close()

def plot_and_save_metric_from_N_agent(entries, destination_model):
    directory_path = path.join(".", "models", destination_model, "figures")
    
    assure_directory_exists(destination_model)

    min_x_data_entry = min(entries, key=lambda x: len(x["x_data"]))
    min_x_data_len = len(min(entries, key=lambda x: len(x["x_data"]))["x_data"])

    x_data = np.array(min_x_data_entry["x_data"])
    y_data = np.array(min_x_data_entry["y_data"])

    max_points = 200
    
    height = 10
    width = 10 * ((x_data.shape[0] + max_points - 1) / max_points)

    fig = plt.figure(figsize=(width, height), dpi=100)

    for entry in entries:
        x_data = np.array(entry["x_data"])[:min_x_data_len]
        y_data = np.array(entry["y_data"])[:min_x_data_len]

        if entry["mean"]:
            plt.plot(x_data.tolist(), y_data.mean(axis=1).tolist(), "--o", label=entry["model_name"])
        
        # if entry["std"]:
        #     y_mean = y_data.mean(axis=1)
        #     y_std = y_data.std(axis=1)
        #     # plt.plot(x_data, y_mean - y_std, alpha=0.3)
        #     # plt.plot(x_data, y_mean + y_std, alpha=0.3)
        #     plt.fill_between(x_data, y_mean - y_std, y_mean + y_std, alpha=0.3)

    plt.title("Average " + entries[0]["title"].lower())
    plt.xlabel(entries[0]["xlabel"])
    plt.ylabel(entries[0]["ylabel"])

    plt.legend()
    plt.tight_layout()
    plt.savefig(path.join(directory_path, entries[0]["title"]))
    plt.close()

if __name__ == "__main__":
    if len(argv) != 2 and len(argv) != 5:
        print("Usage: python plotting.py {model_name}")
        exit(1)

    ploting_data_path = path.join(".", "models", argv[1], "figures", "ploting_data.json")


    if not path.exists(ploting_data_path):
        print(f"Ploting data for model: {argv[1]} doesn't exist")
        exit(1)

    if len(argv) == 5:
        ploting_data_path1 = path.join(".", "models", argv[1], "figures", "ploting_data.json")
        ploting_data_path2 = path.join(".", "models", argv[2], "figures", "ploting_data.json")
        ploting_data_path3 = path.join(".", "models", argv[3], "figures", "ploting_data.json")
        if not path.exists(ploting_data_path):
            print(f"Ploting data for model: {argv[2]} doesn't exist")
            exit(1)

        if not path.exists(ploting_data_path):
            print(f"Ploting data for model: {argv[3]} doesn't exist")
            exit(1)

    # nazwa modelu 1, nazwa modelu 2, nazwa modelu 3, gdzie wsadzamy
    if len(argv) == 5:
        with plt.style.context('fivethirtyeight'):
            ploting_data = [None for i in range(3)]

            with open(ploting_data_path1, "r", encoding='utf-8') as f:
                ploting_data[0] = json.load(f)

            with open(ploting_data_path2, "r", encoding='utf-8') as f:
                ploting_data[1] = json.load(f)
                
            with open(ploting_data_path3, "r", encoding='utf-8') as f:
                ploting_data[2] = json.load(f)
            
            for i in ploting_data[0]:
                i["model_name"] = argv[1]

            for i in ploting_data[1]:
                i["model_name"] = argv[2]

            for i in ploting_data[2]:
                i["model_name"] = argv[3]

            for e1, e2, e3 in zip(*ploting_data):
                plot_and_save_metric_from_N_agent((e1, e2, e3), argv[4])
    
    with plt.style.context('fivethirtyeight'):
        with open(ploting_data_path, "r", encoding='utf-8') as f:
            ploting_data = json.load(f)
            
        for entry in ploting_data:
            plot_and_save_metric(entry, argv[1])