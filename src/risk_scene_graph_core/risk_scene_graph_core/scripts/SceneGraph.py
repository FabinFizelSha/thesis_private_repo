import ast
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from risk_scene_graph_core.config.parameters import img_size, margin, K_rgb, appearance_threshold
from risk_scene_graph_core.scripts.PositionCalculator import PositionCalculator
from matplotlib.patches import Patch

class NodeLayer(Enum):
    OBJECT = 1      
    HUMAN = 2       
    ROOM = 3        
    BUILDING = 4  
    ROBOT = 5  
    
    @staticmethod
    def from_string(value):
        """Convert a string to a NodeLayer enum.

        Args:
            value: String or NodeLayer value.

        Returns:
            Matching NodeLayer or None if not found.
        """
        if isinstance(value, NodeLayer):
            return value
        value_lower = str(value).lower()
        for layer in NodeLayer:
            if layer.name.lower() == value_lower:
                return layer
        return None
    
    def __str__(self):
        """Return the enum name as a string.

        Args:
            None

        Returns:
            Enum name string.
        """
        return self.name

class SceneGraph:
    def __init__(self):
        """Initialize an empty scene graph with a position calculator.

        Args:
            None

        Returns:
            None
        """
        self.scene_graph = nx.DiGraph()
        self.pos_calculator = PositionCalculator()
        self.robot_id = -1  # Fixed ID for robot node

    def update_robot_position(self, tx, rotM=None):
        """Create or update the robot node with the current position.

        Args:
            tx: Translation vector.
            rotM: Optional rotation matrix (unused).

        Returns:
            None
        """
        tx_arr = np.asarray(tx, dtype=float)
        
        if self.robot_id not in self.scene_graph:
            self.scene_graph.add_node(
                self.robot_id,
                id=self.robot_id,
                name="robot",
                meta="Robot/Sensor platform",
                pos=tx_arr.tolist(),
                layer=NodeLayer.ROBOT.name
            )
        else:
            self.scene_graph.nodes[self.robot_id]["pos"] = tx_arr.tolist()

        self._update_robot_proximity_edges()

    def _update_robot_proximity_edges(self, max_objects: int = 3):
        """Link robot to nearest room and up to N nearest objects.

        Args:
            max_objects: Max number of nearest objects to connect.

        Returns:
            None
        """
        if self.robot_id not in self.scene_graph:
            return

        robot_pos = self._parse_pos_if_string(self.scene_graph.nodes[self.robot_id].get("pos"))
        if robot_pos is None:
            return

        robot_arr = np.asarray(robot_pos, dtype=float)
        if not np.isfinite(robot_arr).all():
            return

        # Remove previous proximity edges from robot to keep them updated each frame.
        for _, dst in list(self.scene_graph.out_edges(self.robot_id)):
            self.scene_graph.remove_edge(self.robot_id, dst)

        room_candidates = []
        object_candidates = []

        for node_id, attrs in self.scene_graph.nodes(data=True):
            if node_id == self.robot_id:
                continue

            layer = NodeLayer.from_string(attrs.get("layer", NodeLayer.OBJECT.name))
            if layer not in {NodeLayer.ROOM, NodeLayer.OBJECT}:
                continue

            pos = self._parse_pos_if_string(attrs.get("pos"))
            if pos is None:
                continue

            node_arr = np.asarray(pos, dtype=float)
            if not np.isfinite(node_arr).all():
                continue

            distance = float(np.linalg.norm(node_arr - robot_arr))
            if layer == NodeLayer.ROOM:
                room_candidates.append((distance, node_id))
            else:
                object_candidates.append((distance, node_id))

        room_candidates.sort(key=lambda item: item[0])
        object_candidates.sort(key=lambda item: item[0])

        if room_candidates:
            dist, room_id = room_candidates[0]
            self.scene_graph.add_edge(self.robot_id, room_id, rel="in", distance=round(dist, 4))

        for dist, obj_id in object_candidates[:max_objects]:
            self.scene_graph.add_edge(self.robot_id, obj_id, rel="near", distance=round(dist, 4))
    
    def add_robot_info(self, info_dict):
        """Add or update robot metadata fields on the robot node.

        Args:
            info_dict: Dictionary of robot metadata.

        Returns:
            None
        """
        if self.robot_id not in self.scene_graph:
            self.scene_graph.add_node(
                self.robot_id,
                id=self.robot_id,
                name="robot",
                meta="Robot platform",
                pos=[0, 0, 0],
                layer=NodeLayer.ROBOT.name
            )
        
        for key, value in info_dict.items():
            if key not in ["id", "name", "layer"]:
                self.scene_graph.nodes[self.robot_id][key] = value

    def _calculate_edge_distance(self, src_id, dst_id):
        """Compute Euclidean distance between two nodes, if possible.

        Args:
            src_id: Source node ID.
            dst_id: Destination node ID.

        Returns:
            Distance in meters rounded to 4 decimals, or None.
        """
        try:
            src_node = self.scene_graph.nodes.get(src_id)
            dst_node = self.scene_graph.nodes.get(dst_id)
            
            if src_node is None or dst_node is None:
                return None
            
            src_pos = self._parse_pos_if_string(src_node.get("pos"))
            dst_pos = self._parse_pos_if_string(dst_node.get("pos"))
            
            if src_pos is None or dst_pos is None:
                return None
            
            # Euclidean distance
            src_arr = np.asarray(src_pos, dtype=float)
            dst_arr = np.asarray(dst_pos, dtype=float)

            if not np.isfinite(src_arr).all() or not np.isfinite(dst_arr).all():
                return None
            
            distance = float(np.linalg.norm(dst_arr - src_arr))
            return round(distance, 4)
        except Exception as e:
            return None

    def get_networkx_graph(self):
        """Return the underlying NetworkX graph.

        Args:
            None

        Returns:
            NetworkX DiGraph instance.
        """
        return self.scene_graph

    def process_vlm_update(self, cutout_before, vlm_output):
        """Merge VLM output into the graph with appearance tracking.

        Args:
            cutout_before: Graph cutout sent to VLM.
            vlm_output: Parsed VLM output graph.

        Returns:
            None
        """
        graph = self.scene_graph
        vlm_nodes = vlm_output.get("nodes", [])
        cutout_nodes = cutout_before.get("nodes", [])

        vlm_node_ids = {node["id"] for node in vlm_nodes}
        cutout_node_ids = {node["id"] for node in cutout_nodes}

        for node in vlm_nodes:
            node_id = node["id"]

            if node_id in graph:
                current_count = graph.nodes[node_id].get("appearance", 0)
                graph.nodes[node_id]["appearance"] = current_count + 1
                graph.nodes[node_id].update(node)
            else:
                node_copy = node.copy()
                node_copy["appearance"] = 1
                if "layer" not in node_copy:
                    node_copy["layer"] = NodeLayer.OBJECT.name
                else:
                    layer = NodeLayer.from_string(node_copy.get("layer"))
                    if layer:
                        node_copy["layer"] = layer.name
                graph.add_node(node_id, **node_copy)

        for node_id in (cutout_node_ids - vlm_node_ids):
            if node_id in graph:
                current_count = graph.nodes[node_id].get("appearance", 1)
                new_count = current_count - 1

                if new_count <= appearance_threshold:
                    graph.remove_node(node_id)
                else:
                    graph.nodes[node_id]["appearance"] = new_count

        vlm_edges = vlm_output.get("edges", [])
        cutout_edges = cutout_before.get("edges", [])

        # Update edges with appearance tracking
        vlm_edge_set = set()
        for edge in vlm_edges:
            src, dst = edge["from"], edge["to"]

            if src in graph and dst in graph:
                vlm_edge_set.add((src, dst))
                
                rel = edge.get("rel", "")
                distance = self._calculate_edge_distance(src, dst)
                
                if graph.has_edge(src, dst):
                    # Increment appearance for existing edge
                    current_count = graph[src][dst].get("appearance", 0)
                    graph[src][dst]["appearance"] = current_count + 1
                    graph[src][dst]["rel"] = rel
                    graph[src][dst]["distance"] = distance
                else:
                    # New edge with appearance = 1
                    graph.add_edge(src, dst, rel=rel, distance=distance, appearance=1)

        cutout_edge_set = {(edge["from"], edge["to"]) for edge in cutout_edges}

        # Decrease appearance for edges in cutout but NOT in VLM output
        for src, dst in (cutout_edge_set - vlm_edge_set):
            if graph.has_edge(src, dst):
                current_count = graph[src][dst].get("appearance", 1)
                new_count = current_count - 1

                if new_count <= appearance_threshold:
                    graph.remove_edge(src, dst)
                else:
                    graph[src][dst]["appearance"] = new_count
    
    def visualize_graph(self, save_path="scene_graph.png", save=True, show=False):
        """Render the graph as a PNG image.

        Args:
            save_path: Output file path.
            save: Whether to save the figure.
            show: Whether to display the figure.

        Returns:
            None
        """
        plt.figure(figsize=(16, 10))
        pos_layout = nx.spring_layout(self.scene_graph, seed=42, k=0.5)

        layer_colors = {
            "OBJECT": "lightblue",
            "HUMAN": "lightcoral",
            "ROOM": "lightgreen",
            "BUILDING": "lightyellow",
            "ROBOT": "orange"
        }
        
        node_colors = [layer_colors.get(self.scene_graph.nodes[n].get("layer", "OBJECT"), "lightgray") 
                       for n in self.scene_graph.nodes]

        # Draw nodes and labels
        nx.draw_networkx_nodes(
            self.scene_graph, pos_layout,
            node_size=1000, node_color=node_colors, edgecolors="black"
        )
        labels = {n: self.scene_graph.nodes[n].get("name", str(n)) for n in self.scene_graph.nodes}
        nx.draw_networkx_labels(self.scene_graph, pos_layout, labels=labels, font_size=8)

        # Draw edges and edge labels
        nx.draw_networkx_edges(self.scene_graph, pos_layout, arrows=True, arrowstyle="->")
        edge_labels = {}
        for u, v, d in self.scene_graph.edges(data=True):
            rel = d.get("rel", "")
            distance = d.get("distance")
            if distance is not None:
                edge_labels[(u, v)] = f"{rel}\n({distance}m)"
            else:
                edge_labels[(u, v)] = rel
        nx.draw_networkx_edge_labels(self.scene_graph, pos_layout, edge_labels=edge_labels, font_color="red", font_size=7)

        # Add legend for layers
        legend_elements = [
            Patch(facecolor="lightblue", edgecolor="black", label="OBJECT"),
            Patch(facecolor="lightcoral", edgecolor="black", label="HUMAN"),
            Patch(facecolor="lightgreen", edgecolor="black", label="ROOM"),
            Patch(facecolor="lightyellow", edgecolor="black", label="BUILDING"),
            Patch(facecolor="orange", edgecolor="black", label="ROBOT")
        ]
        plt.legend(handles=legend_elements, loc="upper left", fontsize=10)

        plt.axis("off")
        plt.tight_layout()
        if save:
            plt.savefig(save_path, dpi=300)
            plt.close()
        if show:
            plt.show()

    def _parse_pos_if_string(self, pos):
        """ Parse a stringified list position into a Python list.
            Convert stringified lists like "[0.0, 0.0, 0.0]" to Python lists
        Args:
            pos: Position value or stringified list.

        Returns:
            Parsed list, or original value, or None if invalid.
        """
        if isinstance(pos, str):
            try:
                parsed = ast.literal_eval(pos)
                return parsed
            except Exception:
                return None
        return pos

    def get_cutouts(self, tx, rotM, depth_img=None, depth_tolerance=0.5):
        """Collect visible nodes/edges for the current camera pose.

        Args:
            tx: Translation vector.
            rotM: Rotation matrix.
            depth_img: Optional depth image in meters for frustum cutoff.
            depth_tolerance: Extra range beyond depth values (meters).

        Returns:
            Dictionary with visible "nodes" and "edges".
        """
        visible_objects = {"nodes": [], "edges": []}
        visible_node_ids = set()
        room_candidates = []
        tx_arr = np.asarray(tx, dtype=float)

        for node_id, attr in self.scene_graph.nodes(data=True):
            # Skip ROBOT layer nodes from being sent to VLM
            layer_raw = attr.get("layer", NodeLayer.OBJECT.name)
            layer_enum = NodeLayer.from_string(layer_raw) or NodeLayer.OBJECT
            layer = layer_enum.name
            if layer == NodeLayer.ROBOT.name:
                continue
            
            pos = self._parse_pos_if_string(attr.get("pos"))
            if pos is None:
                continue

            point = np.asarray(pos, dtype=float)
            if layer == NodeLayer.ROOM.name and np.isfinite(point).all():
                distance = float(np.linalg.norm(point - tx_arr))
                room_candidates.append((distance, node_id, attr, pos))

            inside_cut = self.is_probably_visible(point, tx_arr, rotM, depth_img, depth_tolerance)
            if inside_cut:
                visible_objects["nodes"].append({
                    "id": node_id,
                    "name": attr.get("name", str(node_id)),
                    "semantic_description": attr.get("semantic_description", ""),
                    "semantic_pose": attr.get("semantic_pose", ""),
                    "risk": attr.get("risk", ""),
                    "pos": pos, 
                    "confidence": attr.get("confidence", 0.0),
                    "gt": bool(attr.get("gt", True)),
                    "layer": layer
                })
                visible_node_ids.add(node_id)

        if room_candidates:
            room_candidates.sort(key=lambda item: item[0])
            _, room_id, room_attr, room_pos = room_candidates[0]
            if room_id not in visible_node_ids:
                visible_objects["nodes"].append({
                    "id": room_id,
                    "name": room_attr.get("name", str(room_id)),
                    "semantic_description": room_attr.get("semantic_description", ""),
                    "semantic_pose": room_attr.get("semantic_pose", ""),
                    "risk": room_attr.get("risk", ""),
                    "pos": room_pos,
                    "confidence": room_attr.get("confidence", 0.0),
                    "gt": False,
                    "layer": NodeLayer.ROOM.name
                })
                visible_node_ids.add(room_id)

        for src, dst, edata in self.scene_graph.edges(data=True):
            if src in visible_node_ids and dst in visible_node_ids:
                visible_objects["edges"].append({
                    "from": src, "to": dst, "rel": edata.get("rel", ""), "distance": edata.get("distance")
                })

        return visible_objects

    def is_probably_visible(self, point, origin, rotM, depth_img=None, depth_tolerance=0.1):
        """Check if a 3D point is within the camera's field of view and range.

        Args:
            point: 3D point in world coordinates.
            origin: Camera origin in world coordinates.
            rotM: Rotation matrix (camera to world).
            depth_img: Optional depth image in meters for range cutoff.
            depth_tolerance: Extra range beyond depth values (meters).

        Returns:
            True if point is likely visible, else False.
        """
        point = np.asarray(point, dtype=float)
        origin = np.asarray(origin, dtype=float)

        fx, fy = K_rgb[0, 0], K_rgb[1, 1]
        height, width = img_size  

        fov_x = 2 * np.degrees(np.arctan(width / (2 * fx)))
        fov_y = 2 * np.degrees(np.arctan(height / (2 * fy)))

        local = rotM.T @ (point - origin)
        if local[2] <= 0:
            return False

        angle_x = np.degrees(np.arctan2(abs(local[0]), local[2]))
        angle_y = np.degrees(np.arctan2(abs(local[1]), local[2]))
        in_fov = (angle_x < (fov_x / 2) * margin) and (angle_y < (fov_y / 2) * margin)
        if not in_fov:
            return False

        if depth_img is None:
            return True

        px = self.pos_calculator.pos_to_pixel(point[0], point[1], point[2], rotM, origin)
        px_x, px_y = int(px[0]), int(px[1])
        if px_x < 0 or px_y < 0:
            return False

        height, width = depth_img.shape[:2]
        if px_x >= width or px_y >= height:
            return False

        depth_val = float(depth_img[px_y, px_x])
        if depth_val <= 0 or np.isnan(depth_val):
            return True

        return local[2] <= depth_val + float(depth_tolerance)

    def convert_graph_pos_to_pixel(self, graph, tx, rotM):
        """Convert node 3D positions to normalized pixel coordinates.

        Args:
            graph: Graph dictionary with node positions.
            tx: Translation vector.
            rotM: Rotation matrix.

        Returns:
            Updated graph dictionary with x/y fields.
        """
        if graph is None:
            return {"nodes": [], "edges": []}

        height, width = img_size

        for node in graph["nodes"]:
            pos = node["pos"]
            px = self.pos_calculator.pos_to_pixel(pos[0], pos[1], pos[2], rotM, tx)
            x_pix, y_pix = float(px[0]), float(px[1])

            # Clamp to pixel bounds
            x_clamped = max(0, min(x_pix, width - 1))
            y_clamped = max(0, min(y_pix, height - 1))

            # GT: true if no clamping occurred (compare in pixel space)
            gt_flag = (x_pix == x_clamped) and (y_pix == y_clamped)

            # Convert to relative [0,1] and limit precision for VLM input
            x_rel = round(x_clamped / (width - 1), 2)
            y_rel = round(y_clamped / (height - 1), 2)

            del node["pos"]
            node["x"] = x_rel
            node["y"] = y_rel
            node["gt"] = gt_flag

        return graph

    def convert_graph_pixel_to_pos(self, graph, depth_img, tx, rotM):
        """Convert node pixel coordinates to 3D positions using depth.

        Args:
            graph: Graph dictionary with node pixel coords.
            depth_img: Depth image in meters.
            tx: Translation vector.
            rotM: Rotation matrix.

        Returns:
            Updated graph dictionary with pos fields.
        """
        height, width = img_size

        for node in graph["nodes"]:
            x_rel = node.get("x")
            y_rel = node.get("y")

            x_pix = int(round(x_rel * (width - 1)))
            y_pix = int(round(y_rel * (height - 1)))

            x_clipped = max(0, min(x_pix, width - 1))
            y_clipped = max(0, min(y_pix, height - 1))

            if x_pix != x_clipped or y_pix != y_clipped:
                node["gt"] = False

            pos = self.pos_calculator.pixel_to_pos(x_clipped, y_clipped, depth_img, rotM, tx)

            del node["x"]
            del node["y"]
            node["pos"] = pos

        return graph

