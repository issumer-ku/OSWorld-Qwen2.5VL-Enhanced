"""Small OSWorld accessibility bridge used by the enhanced adapter."""

import xml.etree.ElementTree as ET

import tiktoken


ATTR_NS = "https://accessibility.windows.example.org/ns/attributes"
STATE_NS_UBUNTU = "https://accessibility.ubuntu.example.org/ns/state"
STATE_NS_WINDOWS = "https://accessibility.windows.example.org/ns/state"
COMPONENT_NS_UBUNTU = "https://accessibility.ubuntu.example.org/ns/component"
COMPONENT_NS_WINDOWS = "https://accessibility.windows.example.org/ns/component"
VALUE_NS_UBUNTU = "https://accessibility.ubuntu.example.org/ns/value"
VALUE_NS_WINDOWS = "https://accessibility.windows.example.org/ns/value"
CLASS_NS_WINDOWS = "https://accessibility.windows.example.org/ns/class"


def _helpers():
    try:
        from .heuristic_retrieve import (  # type: ignore
            draw_bounding_boxes,
            filter_nodes,
        )
    except ImportError as exc:
        raise RuntimeError(
            "OSWorld is required for accessibility-tree or SOM observations. "
            "Use the standalone runner with --osworld-root, or put OSWorld on PYTHONPATH."
        ) from exc
    return filter_nodes, draw_bounding_boxes


def linearize_accessibility_tree(accessibility_tree: str, platform: str = "ubuntu") -> str:
    if platform not in {"ubuntu", "windows"}:
        raise ValueError("platform must be 'ubuntu' or 'windows'")
    component_ns = COMPONENT_NS_UBUNTU if platform == "ubuntu" else COMPONENT_NS_WINDOWS
    value_ns = VALUE_NS_UBUNTU if platform == "ubuntu" else VALUE_NS_WINDOWS
    filter_nodes, _ = _helpers()
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    rows = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]
    for node in nodes:
        if node.text:
            text = node.text if '"' not in node.text else '"' + node.text.replace('"', '""') + '"'
        elif node.get(f"{{{CLASS_NS_WINDOWS}}}class", "").endswith("EditWrapper") and node.get(
            f"{{{value_ns}}}value"
        ):
            value = node.get(f"{{{value_ns}}}value", "")
            text = value if '"' not in value else '"' + value.replace('"', '""') + '"'
        else:
            text = '""'
        class_value = node.get(
            f"{{{ATTR_NS if platform == 'ubuntu' else CLASS_NS_WINDOWS}}}class", ""
        )
        rows.append(
            "\t".join(
                [
                    node.tag,
                    node.get("name", ""),
                    text,
                    class_value,
                    node.get(f"{{{ATTR_NS}}}description", ""),
                    node.get(f"{{{component_ns}}}screencoord", ""),
                    node.get(f"{{{component_ns}}}size", ""),
                ]
            )
        )
    return "\n".join(rows)


def tag_screenshot(screenshot, accessibility_tree: str, platform: str = "ubuntu"):
    filter_nodes, draw_bounding_boxes = _helpers()
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform=platform, check_image=True)
    marks, drew_nodes, element_list, tagged = draw_bounding_boxes(nodes, screenshot)
    return marks, drew_nodes, tagged, element_list


def trim_accessibility_tree(tree: str, max_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(tree)
    return tree if len(tokens) <= max_tokens else encoding.decode(tokens[:max_tokens]) + "[...]\n"
