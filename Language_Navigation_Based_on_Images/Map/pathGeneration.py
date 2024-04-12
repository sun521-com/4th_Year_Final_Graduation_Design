import heapq
import networkx as nx
from matplotlib import pyplot as plt


class DirectedGraph:
    """
    A class for creating and managing a directed graph structure.
    """
    def __init__(self):
        self.graph = {}

    def add_vertex(self, vertex_id, image_path=None, is_junction=False, extra_info=None):
        """
        Adds a new vertex to the graph.

        Parameters:
            vertex_id (str): Unique identifier for the vertex.
            image_path (str, optional): Path to an image associated with the vertex.
            is_junction (bool, optional): Flag indicating whether the vertex represents a junction.
            extra_info (dict, optional): Additional information related to the vertex.
        """
        if vertex_id not in self.graph:
            self.graph[vertex_id] = {
                'edges': [],
                'image_path': image_path,
                'is_junction': is_junction,
                'extra_info': extra_info
            }

    def add_edge(self, from_vertex, to_vertex, direction, weight):
        """
        Adds a directed edge between two vertices in the graph.

        Parameters:
            from_vertex (str): The ID of the source vertex.
            to_vertex (str): The ID of the destination vertex.
            direction (str): The direction of the edge (e.g., 'forward', 'back', 'left', 'right').
            weight (float): The weight of the edge, representing cost, distance, or any other metric.
        """
        if from_vertex in self.graph and to_vertex in self.graph:
            self.graph[from_vertex]['edges'].append({
                'to': to_vertex,
                'direction': direction,
                'weight': weight
            })
        else:
            raise ValueError("One or both vertices not found in graph")

    def construct_graph(self, vertices_info, edges_info):
        # add vertices
        for vertex_info in vertices_info:
            vertex_id = vertex_info[0]
            image_path = vertex_info[1]
            is_junction = vertex_info[2]
            extra_info = vertex_info[3] if len(vertex_info) > 3 else None  # Extract extra info if available
            self.add_vertex(vertex_id, image_path, is_junction, extra_info)

        # add edges
        for from_vertex, to_vertex, direction, weight in edges_info:
            self.add_edge(from_vertex, to_vertex, direction, weight)

    def convert_directed_graph_to_dict(self,directed_graph):
        """
        Converts the graph structure into a dictionary format.

        Parameters:
            directed_graph (DirectedGraph): The DirectedGraph instance to convert.

        Returns:
            dict: A dictionary representation of the graph.
        """
        graph_dict = {}
        # Iterate over the nodes in the DirectedGraph object
        for node_id, node_data in directed_graph.graph.items():
            # For each node, copy its information into the new dictionary
            graph_dict[node_id] = {
                'edges': node_data['edges'],
                'image_path': node_data['image_path'],
                'is_junction': node_data['is_junction'],
                'extra_info': node_data['extra_info']
            }
        return graph_dict

    def find_shortest_path(self, graph, start_vertex, target_vertex):
        """
        Finds the shortest path from a start vertex to a target vertex using Dijkstra's algorithm.

        Parameters:
            graph (dict): The graph represented as a dictionary.
            start_vertex (str): The starting vertex ID.
            target_vertex (str): The target vertex ID.

        Returns:
            dict: A dictionary containing the shortest path and related information.
        """
        if start_vertex == target_vertex:
            # It is assumed that when the start and end points are the same,
            # there is no movement and therefore no direction, which can be adjusted to suit actual needs
            return {'current location':start_vertex,'destination':target_vertex,'direction':'no more direction','next stop':'arrived'}

        # Initialize all distances to infinity and set the distance to the start vertex to zero.
        distances = {vertex: float('infinity') for vertex in graph}
        distances[start_vertex] = 0

        # Maintain a dictionary to track the previous node and direction for each vertex.
        previous_nodes = {vertex: None for vertex in graph}
        pq = [(0, start_vertex)]

        while pq:
            # Pop the vertex with the smallest distance from the priority queue.
            current_distance, current_vertex = heapq.heappop(pq)

            # If the target vertex is reached, stop the loop.
            if current_vertex == target_vertex:
                break

            for edge in graph[current_vertex]['edges']:
                neighbor = edge['to']
                weight = edge['weight']
                new_distance = current_distance + weight

                # If the new distance is smaller, update the distance and previous node/direction.
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = (current_vertex, edge['direction'])
                    heapq.heappush(pq, (new_distance, neighbor))

        # Reconstruct the shortest path and directions from start_vertex to target_vertex.
        path = []
        directions = []
        current_vertex = target_vertex

        # Return an empty response if the target is unreachable.
        if distances[target_vertex] == float('infinity'):
            return {[], [],target_vertex}

        # Trace the path from target to start using the previous_nodes map.
        while current_vertex is not None:
            if previous_nodes[current_vertex]:
                pred_vertex, direction = previous_nodes[current_vertex]
                path.append(pred_vertex)
                directions.append(direction)
                current_vertex = pred_vertex
            else:
                break

        # Reverse the path and directions to start from the start_vertex.
        path.reverse()
        path.append(target_vertex)
        directions.reverse()

        # Return the shortest path information including the first direction to take and the next stop.
        return {'current location':start_vertex,'destination':target_vertex,'direction':directions[0],'next arrive':path[1]}


if __name__ == '__main__':
    vertices_info = [
        ('Home', 'home_junction.png', True,
         {'junction_coord': (1004, 971), 'forward_exits_coords': (1004, 745), 'left_exit_coords': (353, 971)}),
        ('Moon Street', 'home_straight.png', False),
        ('Heriot-Watt University', 'school.png', False),
        ('Princesses Street', 'home_left.png', False),
        ('Cross Junction', 'junction.png', True,
         {'junction_coord': (1000, 953), 'forward_exits_coords': (1000, 790), 'left_exit_coords': (40, 953),
          'right_exit_coords': (2000, 953)}),
        ('Queen Street', 'junction_straight.png', False),
        ('McDonald\'s', 'McDonald\'s.png', False),
        ('Princes Street', 'junction_left.png', False),
        ('St.James Quarter', 'St.James_Quarter.png', False),
        ('King Street', 'junction_right.png', False),
        ('Sainsbury', 'Sainsbury.png', False),
    ]

    edges_info = [
        ('Home', 'Moon Street', 'forward', 1),
        ('Moon Street', 'Home', 'back', 1),
        ('Moon Street', 'Heriot-Watt University', 'forward', 1),
        ('Heriot-Watt University', 'Moon Street', 'back', 1),
        ('Home', 'Princesses Street', 'left', 1),
        ('Princesses Street', 'Home', 'back', 1),
        ('Princesses Street', 'Cross Junction', 'forward', 1),
        ('Cross Junction', 'Princesses Street', 'back', 1),
        ('Cross Junction', 'Queen Street', 'forward', 1),
        ('Queen Street', 'Cross Junction', 'back', 1),
        ('Cross Junction', 'Princes Street', 'left', 1),
        ('Princes Street', 'Cross Junction', 'back', 1),
        ('Cross Junction', 'King Street', 'right', 1),
        ('King Street', 'Cross Junction', 'back', 1),
        ('Queen Street', 'McDonald\'s', 'forward', 1),
        ('McDonald\'s', 'Queen Street', 'back', 1),
        ('Princes Street', 'St.James Quarter', 'forward', 1),
        ('St.James Quarter', 'Princes Street', 'back', 1),
        ('King Street', 'Sainsbury', 'forward', 1),
        ('Sainsbury', 'King Street', 'back', 1),
        # 以此类推
    ]

    directedGraph = DirectedGraph()
    directedGraph.construct_graph(vertices_info, edges_info)
    graph = directedGraph.convert_directed_graph_to_dict(directedGraph)
    # 创建图
    G = nx.DiGraph()

    # 添加节点和边
    for node, data in graph.items():
        G.add_node(node, image_path=data['image_path'], is_junction=data['is_junction'])
        for edge in data['edges']:
            if edge['direction'] != 'back':
                G.add_edge(node, edge['to'], weight=edge['weight'], direction=edge['direction'])

    # 绘制图形
    plt.figure(figsize=(20, 10))  # 您可以根据需要调整画布的尺寸
    # pos = nx.spring_layout(G)  # 生成一个布局
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'name')  # 获取所有节点的'name'属性作为标签
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='k', linewidths=1, font_size=10)

    # 为边添加权重注解
    edge_labels = {(u, v): d['direction'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # 显示图形
    plt.show()