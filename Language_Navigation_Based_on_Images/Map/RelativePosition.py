import cv2
from pathGeneration import DirectedGraph
from Detector import Detector


class RelativePosition:
    def __init__(self):
        # 分析每个物体的位置
        self.objects_left_of_junction = []
        self.objects_right_of_junction = []
        self.junction_coord = None
        self.forward_exit_coord = None
        self.left_exit_coord = None
        self.left_exit_detected = False
        self.right_exit_coord = None
        self.right_exit_detected = False

    def parameterAssignment(self, current_node):
        """
        Assigns parameters based on the current node's information, particularly regarding junction coordinates and exits.

        Parameters:
            current_node (dict): The current node containing information about the junction and exits.
        """
        self.junction_coord = current_node['extra_info']['junction_coord']  # Junction's coordinates.
        self.forward_exit_coord = current_node['extra_info']['forward_exits_coords']  # 前进出口的坐标
        if 'left_exit_coords' in current_node['extra_info']:
            self.left_exit_detected = True
            self.left_exit_coord = current_node['extra_info']['left_exit_coords']  # Coordinates of former imports and exports
        if 'right_exit_coords' in current_node['extra_info']:
            self.right_exit_detected = True
            self.right_exit_coord = current_node['extra_info']['right_exit_coords']  # Coordinates of the right exit

    def is_left_of_line(self, p, line_start, line_end):
        """
        Determines if a point is to the left of a line segment.

        Parameters:
            p (tuple): The point to check.
            line_start (tuple): The starting point of the line segment.
            line_end (tuple): The ending point of the line segment.

        Returns:
            bool: True if the point is to the left of the line segment; False otherwise.
        """
        return ((line_end[0] - line_start[0]) * (p[1] - line_start[1]) -
                (line_end[1] - line_start[1]) * (p[0] - line_start[0])) < 0

    def get_object_center(self, bbox):
        """
        Calculates the center point of a bounding box.

        Parameters:
            bbox (list/tuple): The bounding box with coordinates (x_min, y_min, x_max, y_max).

        Returns:
            tuple: The center point (x_center, y_center) of the bounding box.
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        return (x_center, y_center)

    def is_in_front_of_line(self, p, line_start, line_end):
        """
        Determines if a point is in front of a line segment based on the y-coordinate.If the centre of the object is outside the y-coordinate range of line_start and line_end, we consider it to be "in front".
        The logic here depends on the image coordinate system, this example applies when the y-coordinate is increasing downwards in the image.

        Parameters:
            p (tuple): The point to check.
            line_start (tuple): The starting point of the line segment.
            line_end (tuple): The ending point of the line segment.

        Returns:
            bool: True if the point is in front of the line segment; False otherwise.
        """
        return p[1] < min(line_start[1], line_end[1])

    def positionClassification(self, detected_objects):
        """
        Classifies detected objects as being to the left or right of the junction based on their positions.

        Parameters:
            detected_objects (list): A list of objects detected, each described by a bounding box and other properties.
        """
        for obj in detected_objects:
            obj_center = self.get_object_center(obj['BBox'])
            if self.is_left_of_line(obj_center, self.junction_coord, self.forward_exit_coord):
                self.objects_left_of_junction.append(obj)
            else:
                self.objects_right_of_junction.append(obj)

    def turn_description(self, turn_direction):
        """
        Generates a turn description based on the relative positions of objects to the junction.

        Parameters:
            turn_direction (str): The direction of the turn ('left' or 'right').

        Returns:
            str: A description of how to execute the turn based on nearby objects.
        """

        # Initialize variables to hold descriptions and lists of objects ahead of or behind the junction.
        description = None
        objects_ahead = []
        objects_behind = []

        # Determine the relevant objects and exit coordinates based on the turn direction.
        relevant_objects = self.objects_left_of_junction if turn_direction == 'left' else self.objects_right_of_junction
        exit_coord = self.left_exit_coord if turn_direction == 'left' else self.right_exit_coord

        # Classify each relevant object as being ahead of or behind the junction.
        for obj in relevant_objects:
            obj_center = self.get_object_center(obj['BBox'])
            if self.is_in_front_of_line(obj_center, self.junction_coord, exit_coord):
                objects_ahead.append(obj)
            else:
                objects_behind.append(obj)

        # If there are objects ahead of the junction, use the nearest one to construct the turn instruction.
        if objects_ahead:
            # Find the nearest object ahead by comparing the y-coordinates relative to the junction_coord.
            nearest_object_ahead = min(objects_ahead, key=lambda obj: abs(
                self.get_object_center(obj['BBox'])[1] - self.junction_coord[1]))
            # Construct the turn instruction using the nearest object ahead.
            description = f"Make a {turn_direction} turn just before {nearest_object_ahead['class']}. Color: {nearest_object_ahead['color']}, Location: {nearest_object_ahead['location']}"
        elif objects_behind:
            # If there are no objects ahead, use the nearest object behind to construct the turn instruction.
            nearest_object_behind = min(objects_behind, key=lambda obj: abs(
                self.get_object_center(obj['BBox'])[1] - self.junction_coord[1]))
            # Construct the turn instruction using the nearest object behind.
            description = f"Make a {turn_direction} turn just after {nearest_object_behind['class']}. Color: {nearest_object_behind['color']}, Location: {nearest_object_behind['location']}"

        # Return the constructed turn description. If no objects are relevant, provide a clear route instruction.
        return description if description else f"Clear {turn_direction} turn route with no nearby obstacles."


if __name__ == '__main__':
    vertices_info = [
        ('A', 'test/junction1.png', True,
         {'junction_coord': (628, 800), 'forward_exits_coords': (628, 604), 'left_exit_coords': (15, 810)}),
        ('B', 'test/junction1_forward.png', False),
        ('C', 'test/junction1_left.png', False),
        ('D', 'test/4.png', False),
    ]

    edges_info = [
        ('A', 'B', 'forward', 1),
        ('B', 'A', 'back', 1),
        ('A', 'C', 'left', 1),
        ('C', 'A', 'back', 1),
        ('B', 'D', 'forward', 1),
        ('D', 'B', 'back', 1),
        # 以此类推
    ]

    detector = Detector()
    image = cv2.imread("static/test/junction1.png")
    list = detector.resultToList(image)
    # print(list)

    directedGraph = DirectedGraph()
    directedGraph.construct_graph(vertices_info, edges_info)
    graph = directedGraph.convert_directed_graph_to_dict(directedGraph)

    relative_position = RelativePosition()
    current_node = graph['A']
    relative_position.parameterAssignment(current_node)
    relative_position.positionClassification(list)
    print(relative_position.turn_description('left'))
