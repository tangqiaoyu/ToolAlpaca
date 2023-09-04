import os
import json
import requests
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-num", type=int, default=2000)
    args = parser.parse_args()

    if os.path.exists(args.api_data_path):
        api_data = json.load(open(args.api_data_path))
    else:
        public_apis = requests.get("https://api.publicapis.org/entries").json()["entries"]
        api_data = []
        for api in public_apis:
            tmp = {
                "Name": api["API"],
                "Description": api["Description"],
                "Link": api["Link"],
                "Category": api["Category"]
            }
            api_data.append(tmp)
    
    api_data = api_data[:args.num]
    json.dump(
        api_data,
        open(args.api_data_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4
    )
