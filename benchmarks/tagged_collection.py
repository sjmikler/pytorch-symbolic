#  Copyright (c) 2022 Szymon Mikler

import json
from typing import Callable

import numpy as np


def flatten_nested_dict(data):
    for key, value in list(data.items()):
        if isinstance(value, dict):
            value = flatten_nested_dict(value)
            value = {str(key) + "." + str(k): v for k, v in value.items()}
            data.pop(key)
            data.update(value)
    return data


class Tagged(dict):
    def __init__(self, tags=None, featured=None, **fields):
        flatten_fields = flatten_nested_dict(fields)
        self.featured = featured

        if tags is not None:
            self._tags = tags
        else:
            if "tags" in flatten_fields:
                self._tags = flatten_fields.pop("tags")
            elif "data.tags" in flatten_fields:
                self._tags = flatten_fields.pop("data.tags")
            else:
                raise KeyError("Tags are missing!")

        super().__init__(**fields)
        self._tags_str = ",".join(self.tags)
        self["tags"] = self._tags_str

    @property
    def tags(self):
        return self._tags

    def __repr__(self):
        if self.featured:
            details = "; " + str({k: self[k] for k in self.featured})
        else:
            details = ""

        return f"Tagged {self.tags}{details}"


class TaggedCollection(list):
    def __init__(self, data=None, featured=None):
        super().__init__()
        self._tags = []
        self._tags_set = set()
        self.featured = featured

        if data:
            for row in data:
                self.convert_and_append(row)

    def convert_and_append(self, row):
        if not isinstance(row, Tagged):
            row = Tagged(**row, featured=self.featured)

        for tag in row.tags:
            if tag not in self._tags_set:
                self._tags.append(tag)
                self._tags_set.add(tag)
        self.append(row)

    def get(self, key):
        values = []
        for record in self:
            assert key in record
            values.append(record[key])
        return np.array(values)

    @property
    def tags(self):
        return self._tags

    def filter(self, *filters):
        new = TaggedCollection()
        for record in self:
            for flt in filters:
                if not flt(record):
                    break
            else:
                new.convert_and_append(record)
        return new

    def restrict_tags(self, tags):
        new = TaggedCollection()
        tags_set = set(tags)

        for record in self:
            for tag in record.tags:
                if tag not in tags_set:
                    break
            else:
                new.convert_and_append(record)
        return new

    def groupby(self, *filters, return_keys=False):
        if not filters:
            filters = [lambda x: True]

        groups = {}
        for record in self:
            results = []
            for flt in filters:
                if isinstance(flt, Callable):
                    result = flt(record)
                else:
                    result = record[flt]
                results.append(result)

            if len(results) > 1:
                results = tuple(results)
            else:
                results = results[0]

            if results not in groups:
                groups[results] = TaggedCollection()
            groups[results].convert_and_append(record)
        if return_keys:
            return list((k, v) for k, v in groups.items())
        else:
            return list(groups.values())

    @classmethod
    def from_dllogs(cls, path, prefix="DLLL", featured=None):
        tc = cls(featured=featured)
        with open(path, "r") as f:
            for line in f.readlines():
                if line.startswith(prefix):
                    line = line[len(prefix) :].strip()
                    data = json.loads(line)
                    tc.convert_and_append(data)
        return tc

    def common_keys(self):
        keys = None
        for idx, row in enumerate(self):
            if keys is None:
                keys = set(row.keys())
            keys = keys.intersection(row.keys())
        return keys

    def to_df(self):
        import pandas as pd

        return pd.DataFrame.from_records(self)

    @property
    def df(self):
        return self.to_df()

    def __getitem__(self, item):
        if isinstance(item, str):
            if item.startswith("!"):
                item = item[1:]
                return self.filter(lambda x: item not in x.tags)
            return self.filter(lambda x: item in x.tags)
        if isinstance(item, Callable):
            return self.filter(item)
        return super().__getitem__(item)
