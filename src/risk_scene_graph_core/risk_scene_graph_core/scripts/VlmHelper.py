import os
import numpy as np
import base64

import openai
import cv2

from risk_scene_graph_core.config.parameters import tmp_dir, vlm_input_format
from risk_scene_graph_core.scripts.FormatConverter import FormatConverter
from risk_scene_graph_core.scripts.WorkerUtils import ensure_dirs

class VLMHelper:

    def __init__(self, format: str = None, save_output: bool = True, small_vlm: bool = False):
        """Initialize the VLM client and response converter.

        Args:
            format: Optional output format override.

        Returns:
            None
        """
        self.save_output = bool(save_output)
        self.small_vlm = bool(small_vlm)
        self.client = openai.OpenAI(
            base_url=os.environ.get('OPENAI_BASE_URL', 'http://0.0.0.0:8000/v1'),
            api_key='foo',
            timeout=10000
        )
        self.models = [x.id for x in self.client.models.list()]
        
        if format is None:
            format = vlm_input_format
        
        self.converter = FormatConverter(format=format)
        

    def vlm_inference(self, cutout_dict, timestamp, image, detected_objects=None, next_node_id=None):
        """Run VLM inference to update the scene graph.

        Args:
            cutout_dict: Current scene graph cutout (nodes/edges).
            timestamp: Frame timestamp string.
            image: RGB image array.
            detected_objects: Optional list of detected objects with bboxes.
            next_node_id: Optional next available node id for new objects.

        Returns:
            Tuple of (parsed_graph, valid_flag).
        """

        cutout_dict = self.convert_numpy(cutout_dict)
        cutout_dict = self.strip_gt(cutout_dict)
        #cutout_dict.pop("edges", None)  # Comment if the edges should also be fed to the VLM. 

        if self.save_output:
            ensure_dirs()
            input_file = os.path.join(tmp_dir, "inVLM", f"{timestamp}{self.converter.extension}")
            self.converter.dump(cutout_dict, input_file)

        output_format_name = self.converter.format.upper()
        format_example = self.converter.get_example()
        cutout_str = self.converter.dumps(cutout_dict)

        # Build detection context
        detection_context = ""
        if detected_objects:
            detection_lines = [
                "\n\nAlso, High-confidence object detections from another vision model:",
                "Please verify these objects carefully and localize them precisely if confirmed:",
            ]
            for obj in detected_objects:
                x1, y1, x2, y2 = obj["bbox"]
                label = obj["label"]
                detection_lines.append(f"- At region ({x1}, {y1}, {x2}, {y2}): {label}")
            detection_context = "\n".join(detection_lines) + "\n"

        next_id_text = ""
        if next_node_id is not None:
            next_id_text = f"- If you ADD new nodes, you MUST start at ID {next_node_id} and increment by 1.\n"

        small_vlm_prompt_rule = ""
        if self.small_vlm:
            small_vlm_prompt_rule = "- Only ADD up to 3 new nodes total.\n"

        prompt_template = """
You are a scene graph expert.

## Verification First:
- For every node in the provided scene graph, confirm that it is clearly represents a visible entity in the image.
- If the entity represented by the node is NOT visible, CHANGE the id to -99 to indicate it is not visible. Do not reuse the old id for any other node.
- This applies to humans, objects, rooms, and building elements.
- Do not pass through nodes unchanged without verifying visibility of the represented entity.
- DO NOT OVERWRITE NODES WITH NEW ENTITIES from the image. Only ADD new nodes at new IDs if a visible ENTITY is missing. Therefore you MUST expand the graph if needed.
- ALL NEW entities MUST be appended with new IDs to the graph, they MUST START at the specified next available ID for any new nodes you add.

## Hard Deletion rule (no discretion)
- For every input node:
    - You MUST choose exactly one:
    - visible and accurate entity in the image -> keep node with same id in the graph(you can adjust name, attributes, position, relations, NOT LAYER)
    - not visible entity -> CHANGE id to -99 (do not reuse the old id for any other node)
- No node may remain unchanged unless it is explicitly verified visible.

Given a single image, first verify whether the proposed scene graph accurately represents the visual content of the image.
If the graph does not match the image, ADD, or ADJUST (change attributes, positions, or relations)
nodes and edges so the scene graph correctly reflects which entities are visible, and what is not visible, in the image.

Provide the final corrected graph strictly as {output_format_name}, describing all visible and not visible entities as nodes and their relations.

Here is the possible scene graph (nodes and edges). Some positions or attributes may be inaccurate and must be corrected if they do not match the entities in the image.
Some of the nodes may NOT appear as entity the image. Do NOT assume a node is present as entity in the image just because it is in the graph. 
{cutout_str}

{detection_context}

Use the format:
{format_example}

## Core Rules (must follow)
- Possible entities are represented by nodes in the scene graph.
- ALWAYS use the same IDs for the same node when the visuals, name/layer, and approximate position of the entity match.
- IDs are the primary identifier for nodes, not names. Do not change the ID of a entity that is clearly visible, even if you adjust its name or position.
- Only if the entity which is represented by the node is clearly not visible or clearly does not match the image should you CHANGE the id to -99.
- If a node is not visible in the image, the ID must be CHANGED to -99.
- Names can be adjusted if the current ones are inaccurate, but IDs MUST be preserved if it is likely the same node.
- NEVER create duplicate nodes for the same node.
- Do NOT add new nodes unless a visible node is missing in the graph.
- If an entity is visible but is inaccurate, ADJUST its node position, attributes, or relations.
{small_vlm_prompt_rule}
- Do not add new room nodes unless the existing room clearly no longer matches the scene.
- Update positions/metadata and relations as needed.
- ALWAYS change the id of nodes to -99 if necessary to maintain accuracy.
- INCLUDE at least one room node, which represents the space, but do not create new room nodes unless the existing one is clearly inaccurate or not visible.

## Node Adjustment & ID Rules (strict)
- Do not change the ID of any node that is clearly visible as an entity in the image. 
- You may adjust its name, position, attributes, and relations, but the layer stays the same unless clearly incorrect.
- Only change the id to -99 if its not visible as an entity in the image.
- If a node is changed to id -99 because it is not visibleas entity, its old ID cannot be reused for any other node.
- Only assign a new ID when adding a completely new node that does not match any existing node in name, layer, or position.
{next_id_text}
- You MUST START new nodes at the specified next available ID and increment by 1 for each new node.
- DO NOT just OVERWRITE existing nodes with new nodes. Only ADD new nodes at new IDs.

## Layer Types (required for each node)
- OBJECT: Physical items, furniture, tools, equipment
- HUMAN: People, humans, persons, dogs, animals
- ROOM: Rooms (represents space only, not surfaces like walls/floor/ceiling)
- BUILDING: Building structures, floors, exterior elements

## Naming Guidelines
- Always use specific, descriptive node names.
- Descriptive naming is for clarity, correction, or highlighting salient attributes, not for creating duplicates.
- Include salient qualifiers when clearly visible:
    - Examples: "red fire extinguisher", "open silver laptop", "wooden dining table", "CO2 gas canister"
- Avoid generic names like "object", "item", "device" when more detail is visible.
- Do not create a new node just because you want to add more detail to the name (e.g., “on white desk”). Instead, adjust the existing nodes name if it is inaccurate or too generic.
- Avoid including other objects context unless it is essential for identification:
    - “red fire extinguisher” -> correct
    - “red fire extinguisher on white desk” -> do NOT make a new node if it already exists

## Semantic Metadata (strict, mandatory)
- Every node MUST include "semantic_description" and "semantic_pose".
- "semantic_description" must be a short, precise visual description of what the entity is (shape, color, material, key parts), not a relation description.
- Prefer examples like: "grey robot dog with four articulated legs" or "red cylindrical fire extinguisher".
- Avoid relation-heavy for this node metadata descriptions like: "next to table", "in front of door", "on left side of room".
- "semantic_pose" must describe the current physical state or posture of the entity (for example: "hanging", "standing", "lying", "walking", "moving", "open", "closed").
- These fields must be evidence-based from the image and must not be empty.

## Human Handling (strict, mandatory)
- All human nodes must be CHANGED to id -99 if there is no clear visual evidence of a human in the image. This is mandatory.
- Keep the same ID if a human is likely the same person, even if position changes.
- Update positions freely if humans moved.
- Use descriptive human names (e.g., “man in blue jacket”, “woman with glasses”).
- Never duplicate a human with the same description.
- If more humans are visible than in the graph -> ADD missing human nodes (up to 3 new nodes max).
- If fewer humans are visible than in the graph -> CHANGE id to -99 to excess human nodes.
- If a human is close to a entity which is represented by a node -> create edge.
- If an edge exists but the human is no longer close to the connected entity -> REMOVE edge.
- If there is no clear visual evidence of a human in the image, CHANGE id to -99 to all human nodes.

## Forbidden Structure & Surface Nodes (strict)
- NEVER create nodes for visual entities like:
    - Walls, floors, ceilings, windows as structural surfaces
    - Colors or textures of room surfaces (e.g., “white wall”, “black floor”, “wooden ceiling”)
    - Architectural background elements unless they are standalone physical objects (e.g., “door”, “stairs” are allowed if visible)
- ROOM nodes must represent space only - do not convert surfaces into ROOM or OBJECT nodes
- If any such nodes exist in the input scene graph, they MUST be CHANGED to id -99, and their old IDs cannot be reused.
- Do not convert surfaces into OBJECT nodes under any circumstances

## Risk Metadata (strict, mandatory)
- For every visible node, you MUST assess and describe potential safety risk metadata based on what is visually present.
- Use explicit hazard categories when applicable. 
    - examples are: [fire hazard, trip hazard, electrocution hazard, collision hazard, cut/puncture hazard, chemical hazard, biohazard, falling object hazard], but you are NOT limited to this list and should use any other clearly plausible hazard if present.
- Assign a risk severity level for each node: low, medium, or high.
- Risk must be evidence-based from the image. Do not invent hazards that are not visually plausible.
- Do not default to "no hazard" for all nodes.
- Use "no hazard" only when the node has been explicitly checked and no plausible danger is visible.
- Ensure the node risk description clearly contains both hazard type and severity level (for example: "trip hazard (medium)").
- If multiple hazards are present for a node, include the primary hazard first and keep severity explicit.
- Keep focus on this section when generating the risk field and do not override these risk rules elsewhere.

## Additional Constraints
- ALWAYS CHANGE the id of nodes to -99 if they are not visible as an entity in the image.
- Each node must:
    - Have a non-empty semantic_description field
    - Have a non-empty semantic_pose field
    - Have a non-empty risk field that follows the Risk Metadata section above
    - Include a confidence score
    - Have at least one relation
- All nodes must be connected (no isolated nodes).
- Relations must reference valid IDs.
- Before outputting the graph, scan all nodes again and CHANGE the id of any node that is not clearly visible as an entity in the image to -99. Humans, objects, rooms, and buildings must all be verified.
- Ignore any diagnostic/gt flags in the input.

## FINAL SANITY CHECK (MANDATORY BEFORE OUTPUT)
- Ensure all nodes which are not clearly visible as an entity in the image have ID -99
- Ensure no ID was reused, except for ID -99 for invisible entities
- Ensure No node from the graph was deleted without being CHANGED with id -99. Removal is only allowed by CHANGING to id -99.
- Ensure no duplicates exist

## Output Rules
- Output ONLY the completed corrected graph in {output_format_name}.
Do NOT include explanations, comments, or extra text.
"""

        txt = prompt_template.format(
            output_format_name=output_format_name,
            format_example=format_example,
            cutout_str=cutout_str,
            detection_context=detection_context,
            next_id_text=next_id_text,
            small_vlm_prompt_rule=small_vlm_prompt_rule,
        )

        # OpenCV expects BGR for encoding; convert from RGB to keep colors correct.
        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image")

        image_base64 = base64.b64encode(buffer).decode("utf-8")

        data_uri = f"data:image/jpeg;base64,{image_base64}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": txt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }]

        try:
            completion = self.client.chat.completions.create(
                model=os.environ.get('MODEL', self.models[0] if self.models else 'default'),
                messages=messages
            )
        except Exception as e:
            return {}, False

        content = completion.choices[0].message.content
        cleaned, valid = self.converter.parse_from_response(content)
        expected_layers = {}
        input_nodes = cutout_dict.get("nodes") if isinstance(cutout_dict, dict) else None
        if isinstance(input_nodes, list):
            for node in input_nodes:
                if isinstance(node, dict):
                    node_id = node.get("id")
                    layer = node.get("layer")
                    if node_id is not None and layer is not None:
                        expected_layers[node_id] = layer

        cleaned = self.remove_invalid_ids(
            cleaned,
            invalid_ids={-99},
            expected_layers=expected_layers or None,
        )

        if self.save_output:
            ensure_dirs()
            output_file = os.path.join(tmp_dir, "outVLM", f"{timestamp}{self.converter.extension}")
            self.converter.dump(cleaned, output_file)

        
        return cleaned,valid
    
    def classify_object(self, crop_image):
        """Classify a cropped object image with the VLM.

        Args:
            crop_image: Cropped object image (PIL or NumPy).

        Returns:
            Predicted label string or None if unavailable.
        """
        if hasattr(crop_image, 'mode'): 
            crop_np = np.array(crop_image)
        else:
            crop_np = crop_image
        
        prompt = """You are an object classification expert.
            
Look at this cropped object image and provide a single, concise label that describes what this object is.

Guidelines:
- Provide ONLY the object name/label, nothing else
- Use common, simple names (e.g., "chair", "table", "person", "laptop")
- Be specific but concise (e.g., "office chair" not "furniture")
- If the cutout is bad, unclear, or doesn't contain a clear object, respond with just: unknown

Respond with just the label, no explanation."""

        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR))
        if not success:
            return None
        
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{image_base64}"
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }]
        
        try:
            completion = self.client.chat.completions.create(
                model=os.environ.get('MODEL', self.models[0] if self.models else 'default'),
                messages=messages
            )
            
            label = completion.choices[0].message.content.strip()
            label = label.strip('"\'\'').strip().lower()
            
            if not label:
                return None
                
            return label
        except Exception as e:
            return None

    def convert_numpy(self, obj):
        """Recursively convert NumPy types to native Python types.

        Args:
            obj: Arbitrary object potentially containing NumPy types.

        Returns:
            Object with NumPy scalars/arrays converted to native types.
        """
        if isinstance(obj, dict):
            return {k: self.convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def strip_gt(self, obj):
        """Recursively remove any gt fields and distance fields from a nested structure.

        Args:
            obj: Arbitrary object potentially containing gt fields.

        Returns:
            Object with gt fields removed.
        """
        if isinstance(obj, dict):
            return {
                k: self.strip_gt(v)
                for k, v in obj.items()
                if k not in {"gt", "distance", "confidence"}
            }
        if isinstance(obj, list):
            return [self.strip_gt(v) for v in obj]
        return obj

    def remove_invalid_ids(self, graph, invalid_ids=None, expected_layers=None):
        """Remove nodes and edges that reference invalid IDs or violate layer rules.

        Args:
            graph: Scene graph dictionary containing nodes/edges.
            invalid_ids: Set of invalid ID values to remove.
            expected_layers: Optional mapping from node ID to expected layer. If
                provided, nodes whose layer does not match their expected layer
                are removed.

        Returns:
            Cleaned graph dictionary.
        """
        if not isinstance(graph, dict):
            return graph

        invalid_ids = set(invalid_ids or set())
        expected_layers = expected_layers or {}

        nodes = graph.get("nodes")
        valid_node_ids = None
        if isinstance(nodes, list):
            filtered_nodes = []
            for node in nodes:
                if not isinstance(node, dict):
                    filtered_nodes.append(node)
                    continue

                node_id = node.get("id")
                if node_id in invalid_ids:
                    continue

                name = node.get("name")
                if isinstance(name, str) and name.strip() == "":
                    continue

                if node_id in expected_layers:
                    expected = expected_layers.get(node_id)
                    actual = node.get("layer")
                    if isinstance(expected, str) and isinstance(actual, str):
                        if expected.strip().upper() != actual.strip().upper():
                            continue

                filtered_nodes.append(node)
            graph["nodes"] = filtered_nodes

            valid_node_ids = {
                node.get("id")
                for node in filtered_nodes
                if isinstance(node, dict) and node.get("id") is not None
            }

        edges = graph.get("edges")
        if isinstance(edges, list):
            filtered_edges = []
            for edge in edges:
                if isinstance(edge, dict):
                    src = edge.get("from")
                    dst = edge.get("to")
                    if valid_node_ids is not None and (
                        src not in valid_node_ids or dst not in valid_node_ids
                    ):
                        continue
                filtered_edges.append(edge)
            graph["edges"] = filtered_edges

        return graph


class DummyVLMHelper:
    
    def __init__(self, format: str = None, save_output: bool = True, small_vlm: bool = False):
        """Initialize dummy helper with a format converter.

        Args:
            format: Optional output format override.

        Returns:
            None
        """
        self.save_output = bool(save_output)
        self.small_vlm = bool(small_vlm)
        if format is None:
            format = vlm_input_format
        self.converter = FormatConverter(format=format)
    
    def vlm_inference(self, cutout_dict, timestamp, image, detected_objects=None, next_node_id=None):
        """Return input dict unchanged for testing.

        Args:
            cutout_dict: Current scene graph cutout.
            timestamp: Frame timestamp string.
            image: RGB image array.
            detected_objects: Optional list of detected objects.

        Returns:
            Tuple of (cutout_dict, True).
        """
        return cutout_dict, True
    
    def classify_object(self, crop_image):
        """Return a default "unknown" label for testing.

        Args:
            crop_image: Cropped object image.

        Returns:
            The string "unknown".
        """
        return "unknown"




