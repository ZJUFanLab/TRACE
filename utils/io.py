# -*- encoding: utf-8 -*-


from __future__ import annotations, print_function, absolute_import
from typing import Text, Union, List, Any
from pathlib import Path

from ruamel import yaml as yaml

def _is_ascii(text: Text) -> bool:
    return all(ord(character) < 128 for character in text)

def read_file(filename) -> Any:
    with open(filename, encoding="utf-8") as f:
        return f.read()

def read_yaml(content: Text,
              reader_type: Union[Text, List[Text]] = "safe"
              ) -> Any:
    if _is_ascii(content):
        # Required to make sure emojis are correctly parsed
        content = (
            content.encode("utf-8")
                .decode("raw_unicode_escape")
                .encode("utf-16", "surrogatepass")
                .decode("utf-16")
        )

    yaml_parser = yaml.YAML(typ=reader_type)
    yaml_parser.preserve_quotes = True

    return yaml_parser.load(content) or {}


def read_config_yaml(filename: Union[Text, Path]) -> Any:
    content = read_file(filename)
    return read_yaml(content)


def dict_to_yaml(dict_value: dict, allow_unicode:bool=True):
    """dict保存为yaml"""
    return yaml.dump(dict_value, allow_unicode=allow_unicode)


def save_file(content):
    pass
