# TOON support: pip install git+https://github.com/toon-format/toon-python.git
import json
import re
from abc import ABC, abstractmethod
from typing import Any

class FormatHandler(ABC):
    
    @abstractmethod
    def dumps(self, data: Any) -> str:
        """Serialize data to a string.

        Args:
            data: Data to serialize.

        Returns:
            Serialized string.
        """
        pass
    
    @abstractmethod
    def loads(self, data: str) -> Any:
        """Deserialize data from a string.

        Args:
            data: Serialized string.

        Returns:
            Deserialized object.
        """
        pass
    
    @abstractmethod
    def dump(self, data: Any, file_path: str) -> None:
        """Serialize data and write it to a file.

        Args:
            data: Data to serialize.
            file_path: Target file path.

        Returns:
            None
        """
        pass
    
    @abstractmethod
    def load(self, file_path: str) -> Any:
        """Load and deserialize data from a file.

        Args:
            file_path: Source file path.

        Returns:
            Deserialized object.
        """
        pass
    
    @abstractmethod
    def parse_from_response(self, content: str) -> tuple[Any, bool]:
        """Parse a model response into structured data.

        Args:
            content: Raw response string.

        Returns:
            Tuple of parsed data and a success flag.
        """
        pass
    
    @property
    @abstractmethod
    def extension(self) -> str:
        """Get the file extension for this format.

        Args:
            None

        Returns:
            File extension string (including dot).
        """
        pass


class JSONHandler(FormatHandler):
    
    EXAMPLE = """{
        "nodes": [
            {"id": 1, "name": "picture", "semantic_description": "picture with trees and red picture frame", "semantic_pose": "hanging", "risk": "collision hazard (low)", "x": 0.5, "y": 0.5, "confidence": 0.9, "layer": "OBJECT"},
            ...
        ],
        "edges": [
            {"from": 1, "to": 2, "rel": "relation"},
            ...
        ]
        }"""
    
    def dumps(self, data: Any) -> str:
        """Serialize data to JSON string.

        Args:
            data: Data to serialize.

        Returns:
            JSON-formatted string.
        """
        return json.dumps(data, indent=2)
    
    def loads(self, data: str) -> Any:
        """Deserialize JSON string to Python object.

        Args:
            data: JSON string.

        Returns:
            Deserialized object.
        """
        return json.loads(data)
    
    def dump(self, data: Any, file_path: str) -> None:
        """Serialize data to JSON and write to a file.

        Args:
            data: Data to serialize.
            file_path: Target file path.

        Returns:
            None
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, file_path: str) -> Any:
        """Load JSON from a file.

        Args:
            file_path: Source file path.

        Returns:
            Deserialized object.
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def parse_from_response(self, content: str) -> tuple[Any, bool]:
        """Extract JSON object from a raw response string.

        Args:
            content: Raw response string.

        Returns:
            Tuple of parsed data and a success flag.
        """
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return {"raw": content}, False
        try:
            return self.loads(content[start:end+1]), True
        except:
            return {"raw": content[start:end+1]}, False
    
    @property
    def extension(self) -> str:
        """Get the JSON file extension.

        Args:
            None

        Returns:
            ".json".
        """
        return '.json'


class YAMLHandler(FormatHandler):
    
    EXAMPLE = """nodes:
          - id: 1
            name: picture
            semantic_description: picture with trees and red picture frame
            semantic_pose: hanging
            risk: collision hazard (low)
            x: 0.5
            y: 0.5
            confidence: 0.9
            layer: OBJECT
        edges:
          - from: 1
            to: 2
            rel: relation"""
    
    def __init__(self):
        """Initialize YAML handler and import PyYAML.

        Args:
            None

        Returns:
            None
        """
        import yaml
        self.yaml = yaml
    
    def dumps(self, data: Any) -> str:
        """Serialize data to YAML string.

        Args:
            data: Data to serialize.

        Returns:
            YAML-formatted string.
        """
        return self.yaml.dump(data, default_flow_style=False)
    
    def loads(self, data: str) -> Any:
        """Deserialize YAML string to Python object.

        Args:
            data: YAML string.

        Returns:
            Deserialized object.
        """
        return self.yaml.safe_load(data)
    
    def dump(self, data: Any, file_path: str) -> None:
        """Serialize data to YAML and write to a file.

        Args:
            data: Data to serialize.
            file_path: Target file path.

        Returns:
            None
        """
        with open(file_path, 'w') as f:
            self.yaml.dump(data, f, default_flow_style=False)
    
    def load(self, file_path: str) -> Any:
        """Load YAML from a file.

        Args:
            file_path: Source file path.

        Returns:
            Deserialized object.
        """
        with open(file_path, 'r') as f:
            return self.yaml.safe_load(f)
    
    def parse_from_response(self, content: str) -> tuple[Any, bool]:
        """Parse YAML content from a raw response string.

        Args:
            content: Raw response string.

        Returns:
            Tuple of parsed data and a success flag.
        """
        if '```' in content:
            start = content.find('```')
            start = content.find('\n', start) + 1
            end = content.find('```', start)
            if end == -1:
                format_str = content[start:]
            else:
                format_str = content[start:end]
        else:
            format_str = content.strip()
        
        try:
            return self.loads(format_str), True
        except:
            try:
                return self.loads(content), True
            except:
                return {"raw": format_str}, False
    
    @property
    def extension(self) -> str:
        """Get the YAML file extension.

        Args:
            None

        Returns:
            ".yaml".
        """
        return '.yaml'


class TOONHandler(FormatHandler):
    
    EXAMPLE = """nodes[1]{id,name,semantic_description,semantic_pose,risk,x,y,confidence,layer}:
        1,picture,picture with trees and red picture frame,hanging,collision hazard (low),0.5,0.5,0.9,OBJECT
edges[1]{from,to,rel}:
  1,2,relation"""
    
    def __init__(self):
        """Initialize TOON handler and import the encoder.

        Args:
            None

        Returns:
            None
        """

        from toon_format import encode, decode
        self._encode = encode
        self._decode = decode
    
    def dumps(self, data: Any) -> str:
        """Serialize data to TOON string.

        Args:
            data: Data to serialize.

        Returns:
            TOON-formatted string.
        """
        return self._encode(data)
    
    def loads(self, data: str) -> Any:
        """Deserialize TOON string to Python object.

        Args:
            data: TOON string.

        Returns:
            Deserialized object.
        """
        result = self._decode(data)
        return dict(result) if not isinstance(result, dict) else result

    def dump(self, data: Any, file_path: str) -> None:
        """Serialize data to TOON and write to a file.

        Args:
            data: Data to serialize.
            file_path: Target file path.

        Returns:
            None
        """
        with open(file_path, 'w') as f:
            f.write(self.dumps(data))
    
    def load(self, file_path: str) -> Any:
        """Load TOON data from a file.

        Args:
            file_path: Source file path.

        Returns:
            Deserialized object.
        """
        with open(file_path, 'r') as f:
            result = self._decode(f.read())
        return dict(result) if not isinstance(result, dict) else result
    
    def parse_from_response(self, content: str) -> tuple[Any, bool]:
        """Parse TOON content from a raw response string.

        Args:
            content: Raw response string.

        Returns:
            Tuple of parsed data and a success flag.
        """
        format_str = content.replace("\r\n", "\n").replace("\r", "\n")
        if "```" in content:
            start = format_str.find("```")
            start = format_str.find("\n", start) + 1
            end = format_str.find("```", start)
            if end == -1:
                format_str = format_str[start:]
            else:
                format_str = format_str[start:end]
        else:
            idx_nodes = format_str.find("nodes[")
            idx_edges = format_str.find("edges[")
            start_idx = min(i for i in [idx_nodes, idx_edges] if i != -1) if (idx_nodes != -1 or idx_edges != -1) else -1
            if start_idx != -1:
                format_str = format_str[start_idx:]

        format_str = self._normalize_toon_counts(format_str)

        try:
            return self.loads(format_str.strip()), True
        except:
            try:
                return self._decode(format_str.strip(), {"strict": False}), True
            except:
                try:
                    return self.loads(content.strip()), True
                except:
                    return {"raw": format_str.strip()}, False

    def _normalize_toon_counts(self, text: str) -> str:
        lines = text.strip().splitlines()
        if not lines:
            return text

        def find_header(prefix: str) -> int:
            for i, line in enumerate(lines):
                if line.lstrip().startswith(prefix + "["):
                    return i
            return -1

        def count_rows(start_idx: int, end_idx: int) -> int:
            count = 0
            for line in lines[start_idx + 1:end_idx]:
                if not line.strip():
                    continue
                if line.lstrip().startswith("nodes[") or line.lstrip().startswith("edges["):
                    continue
                count += 1
            return count

        nodes_idx = find_header("nodes")
        edges_idx = find_header("edges")

        if nodes_idx != -1:
            nodes_end = edges_idx if edges_idx != -1 else len(lines)
            nodes_count = count_rows(nodes_idx, nodes_end)
            lines[nodes_idx] = re.sub(
                r"^(\s*nodes)\[\d+\](.*)$",
                rf"\1[{nodes_count}]\2",
                lines[nodes_idx],
            )

        if edges_idx != -1:
            edges_end = len(lines)
            edges_count = count_rows(edges_idx, edges_end)
            lines[edges_idx] = re.sub(
                r"^(\s*edges)\[\d+\](.*)$",
                rf"\1[{edges_count}]\2",
                lines[edges_idx],
            )

        return "\n".join(lines)
    
    @property
    def extension(self) -> str:
        """Get the TOON file extension.

        Args:
            None

        Returns:
            ".toon".
        """
        return '.toon'


class FormatConverter:
    
    SUPPORTED_FORMATS = {
        'json': JSONHandler,
        'yaml': YAMLHandler,
        'toon': TOONHandler,
    }
    
    def __init__(self, format: str = 'json'):
        """Initialize the converter with a specific output format.

        Args:
            format: Format name (e.g., "json", "yaml", "toon").

        Returns:
            None
        """
        self._format = None
        self._handler = None
        self.set_format(format)
    
    def set_format(self, format: str) -> None:
        """Set the active format handler.

        Args:
            format: Format name (case-insensitive).

        Returns:
            None
        """
        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            format = 'json'
        self._format = format
        self._handler = self.SUPPORTED_FORMATS[format]()
    
    @property
    def format(self) -> str:
        """Get the current format name.

        Args:
            None

        Returns:
            Current format name.
        """
        return self._format
    
    @property
    def extension(self) -> str:
        """Get the file extension for the current format.

        Args:
            None

        Returns:
            File extension string.
        """
        return self._handler.extension
    
    def dumps(self, data: Any) -> str:
        """Serialize data to the current format string.

        Args:
            data: Data to serialize.

        Returns:
            Serialized string.
        """
        return self._handler.dumps(data)
    
    def dump(self, data: Any, file_path: str) -> None:
        """Serialize data and write to a file in the current format.

        Args:
            data: Data to serialize.
            file_path: Target file path.

        Returns:
            None
        """
        self._handler.dump(data, file_path)
    
    def load(self, file_path: str) -> Any:
        """Load data from a file using the current format.

        Args:
            file_path: Source file path.

        Returns:
            Deserialized object.
        """
        return self._handler.load(file_path)
    
    def get_example(self) -> str:
        """Get a sample document for the current format.

        Args:
            None

        Returns:
            Example string.
        """
        return self._handler.EXAMPLE
    
    def parse_from_response(self, content: str) -> tuple[Any, bool]:
        """Parse a model response using the current format handler.

        Args:
            content: Raw response string.

        Returns:
            Tuple of parsed data and a success flag.
        """
        return self._handler.parse_from_response(content)
