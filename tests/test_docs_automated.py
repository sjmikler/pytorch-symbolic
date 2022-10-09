#  Copyright (c) 2022 Szymon Mikler

import os
import pathlib
import re


def scan_for_code_blobs(text):
    blobs = re.findall(r"```py\n([\s\S]+?)\n```", text)
    return [blob for blob in blobs if "..." not in blob]


def test_all_code_blobs():
    assert os.path.exists("docs")
    all_code_blobs = []

    for root, dirs, files in os.walk("docs"):
        for file in files:
            path = pathlib.Path(os.path.join(root, file))
            if path.suffix == ".md":
                code_blobs = scan_for_code_blobs(path.open("r").read())
                for blob in code_blobs:
                    all_code_blobs.append(blob)

    for idx, blob in enumerate(all_code_blobs):
        try:
            globals_temp = {}
            exec(blob, globals_temp)
        except Exception as e:
            print(f"Exception during automated documentation testing {idx}/{len(all_code_blobs)}:")
            print(blob)
            raise e
