import numpy as np
import math
import yaml
import heapq
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch
from shapely import affinity
import itertools

INF = float("inf")

class Environment:
    def __init__(self, yaml_file=None, bounds=None):
        self.yaml_file = yaml_file
        self.obstacles = []
        # self.obstacles_map = {}
        self.bounds = bounds
        if yaml_file:
            self.load_from_yaml_file(yaml_file)

    def bounds(self):
        return self.bounds

    def add_obstacles(self, obstacles):
        self.obstacles = self.obstacles + obstacles

    def load_from_yaml_file(self, yaml_file):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        return self.parse_yaml_data(data)

    def parse_yaml_data(self, data):
        if 'environment' in data:
            env = data['environment']
            self.parse_yaml_obstacles(env['obstacles'])

    def parse_yaml_obstacles(self, obstacles):
        # only parse rectangle and polygon obstacles
        for name, description in obstacles.items():
            # Double underscore not allowed in region names.
            if name.find("__") != -1:
                raise Exception("Names cannot contain double underscores.")
            if description['shape'] == 'rectangle':
                parsed = self.parse_rectangle(name, description)
            elif description['shape'] == 'polygon':
                parsed = self.parse_polygon(name, description)
            else:
                raise Exception("not a rectangle or polygon")
            if not parsed.is_valid:
                raise Exception("%s is not valid!"%name)
            self.obstacles.append(parsed)
            # self.obstacles_map[name] = parsed

    def parse_rectangle(self, name, description):
        center = description['center']
        center = Point((center[0], center[1]))
        length = description['length']
        width = description['width']
        # convert rotation to radians
        rotation = description['rotation']# * math.pi/180
        # figure out the four corners.
        corners = [(center.x - length/2., center.y - width/2.),
                   (center.x + length/2., center.y - width/2.),
                   (center.x + length/2., center.y + width/2.),
                   (center.x - length/2., center.y + width/2.)]
        # print corners
        polygon = Polygon(corners)
        out = affinity.rotate(polygon, rotation, origin=center)
        out.name = name
        out.cc_length = length
        out.cc_width = width
        out.cc_rotation = rotation
        return out

    def parse_polygon(self, name, description):
        _points = description['corners']
        for points in itertools.permutations(_points):
            polygon = Polygon(points)
            polygon.name = name
            if polygon.is_valid:
                return polygon


class Drawer:
    def __init__(self, env):
        self.env = env

    def plot_environment(self):
        minx, miny, maxx, maxy = self.env.bounds

        max_width, max_height = 5, 5
        figsize = (max_width, max_height)
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(111)
        for i, obs in enumerate(self.env.obstacles):
            patch = PolygonPatch(obs, fc='blue', ec='blue', alpha=0.5, zorder=20)
            ax.add_patch(patch)

        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        ax.set_aspect('equal', adjustable='box')
        # Cancel coordinate axis display
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_many_points(self, env_plot, point_array, pointSize=1, colorselect='gray'):
        x = [point[0] for point in point_array]
        y = [point[1] for point in point_array]
        env_plot.plot(x, y,'o',markersize=pointSize,color=colorselect)

    def plot_many_edges(self, env_plot, edge_array, color='gray', lineWidthSelect=0.8):
        for edge in edge_array:
            x = [point[0] for point in edge]
            y = [point[1] for point in edge]
            env_plot.plot(x, y, color=color, linewidth=lineWidthSelect, solid_capstyle='butt', zorder=1)

    def plot_path(self, env_plot, path, color="red"):
        if path:
            x = [point[0] for point in path]
            y = [point[1] for point in path]
            env_plot.plot(x, y,color=color,linewidth=2)

    def plot_ellipse(self, env_plot, c_best, start, goal, color="blue"):
        if c_best!=INF:
            center_xy = ((start[0]+goal[0])/2, (start[0]+goal[0])/2)
            two_c = math.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2)
            two_b=math.sqrt(c_best**2 - two_c**2)
            angle=math.atan2(start[1]-goal[1],start[0]-goal[0])/math.pi*180

            ellipse = Ellipse(xy=center_xy, width=c_best, height=two_b, angle=angle)

            env_plot.add_artist(ellipse)
            ellipse.set_fill(0)
            ellipse.set_edgecolor(color)
            ellipse.set_linestyle('--')
            ellipse.set_linewidth(1.5)

    def plot_iteration(self, cost_set, label_set):
        for i in range(len(cost_set)):
            X = [item[0] for item in cost_set[i]]
            Y = [item[-1] for item in cost_set[i]]
            plt.plot(X,Y, label=label_set[i])
        plt.legend(loc='upper right')
        plt.xlabel("Number of iterations")
        plt.ylabel("Path Cost")
        plt.show()


class BITStar:
    def __init__(self, environment, start, goal, bounds, maxIter=50000, plot_flag=True):
        self.env = environment
        self.obstacles = environment.obstacles

        self.draw = Drawer(self.env)

        self.start = start
        self.goal = goal

        self.bounds = bounds
        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.dimension = 2

        # This is the tree
        self.vertices = []
        self.edges = dict()     # key = point，value = parent
        self.g_scores = dict()

        self.samples = []
        self.vertex_queue = []
        self.edge_queue = []
        self.old_vertices = set()

        self.maxIter = maxIter
        self.r = INF
        self.batch_size = 200
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        # the parameters for informed sampling
        self.c_min = self.distance(self.start, self.goal)
        self.center_point = None
        self.C = None

        # whether plot the middle planning process
        self.plot_planning_process = plot_flag

    def setup_planning(self):
        # add goal to the samples
        self.samples.append(self.goal)
        self.g_scores[self.goal] = INF

        # add start to the tree
        self.vertices.append(self.start)
        self.g_scores[self.start] = 0

        # Computing the sampling space
        self.informed_sample_init()
        radius_constant = self.radius_init()

        return radius_constant

    def radius_init(self):
        # Hypersphere radius calculation
        unit_ball_volume = math.pi
        n = self.dimension
        gamma = (1.0 + 1.0/n) * (self.maxx - self.minx) * (self.maxy - self.miny)/unit_ball_volume
        radius_constant = 2 * self.eta * (gamma**(1.0/n))
        return radius_constant

    def informed_sample_init(self):
        self.center_point = np.matrix([[(self.start[0] + self.goal[0]) / 2.0],[(self.start[1] + self.goal[1]) / 2.0], [0]])
        a_1 = np.matrix([[(self.goal[0] - self.start[0]) / self.c_min],[(self.goal[1] - self.start[1]) / self.c_min], [0]])
        id1_t = np.matrix([1.0,0,0])
        M = np.dot(a_1, id1_t)
        U,S,Vh = np.linalg.svd(M, 1, 1)
        self.C = np.dot(np.dot(U, np.diag([1.0,1.0,  np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh)

    def sample_unit_ball(self):
        a = np.random.random()
        b = np.random.random()
        if b < a:
            a,b=b,a
        sample = (b*math.cos(2*math.pi*a/b), b*math.sin(2*math.pi*a/b))
        return np.array([[sample[0]], [sample[1]], [0]])

    def informed_sample(self, c_best, sample_num):
        if c_best < float('inf'):
            c_b = math.sqrt(c_best**2 - self.c_min**2)/2.0
            r= [c_best /2.0, c_b, c_b]
            L = np.diag(r)
        sample_array = []
        cur_num = 0
        while cur_num < sample_num:
            if c_best < float('inf'):
                x_ball = self.sample_unit_ball()
                random_point = np.dot(np.dot(self.C,L), x_ball) + self.center_point
                random_point = (random_point[(0,0)], random_point[(1,0)])
                if not self.is_point_free(random_point):
                    continue
            else:
                random_point = self.get_collision_free_random_point()
            cur_num += 1
            sample_array.append(random_point)
        return sample_array

    def get_collision_free_random_point(self):
        # Run until a valid point is found
        while True:
            point = self.get_random_point()
            if self.is_point_free(point):
                return point

    def get_random_point(self):
        x = self.minx + np.random.random() * (self.maxx - self.minx)
        y = self.miny + np.random.random() * (self.maxy - self.miny)
        return (x, y)

    def is_point_free(self, point):
        buffered_point = Point(point).buffer(self.obj_radius+1, self.resolution)
        for obstacle in self.obstacles:
            if obstacle.intersects(buffered_point):
                return False
        return True

    def is_edge_free(self, edge):
        line = LineString(edge)
        expanded_line = line.buffer(self.obj_radius, self.resolution)
        for obstacle in self.obstacles:
            if expanded_line.intersects(obstacle):
                return False
        return True

    def get_g_score(self, point):
        # gT(x)
        if point == self.start:
            return 0
        if point not in self.edges:
            return INF
        else:
            return self.g_scores.get(point)

    def get_f_score(self, point):
        # f^(x)
        return self.heuristic_cost(self.start,point) + self.heuristic_cost(point, self.goal)

    def actual_edge_cost(self, point1, point2):
        # c(x1,x2)
        if not self.is_edge_free([point1,point2]):
            return INF
        return self.distance(point1, point2)

    def heuristic_cost(self, point1, point2):
        # Euler distance as the heuristic distance
        return self.distance(point1, point2)

    def distance(self,point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def get_edge_value(self,edge):
        # sort value for edge
        return self.get_g_score(edge[0])+self.heuristic_cost(edge[0],edge[1])+self.heuristic_cost(edge[1],self.goal)

    def get_point_value(self,point):
        # sort value for point
        return self.get_g_score(point)+self.heuristic_cost(point,self.goal)

    def bestVertexQueueValue(self):
        if not self.vertex_queue:
            return INF
        else:
            return self.vertex_queue[0][0]

    def bestEdgeQueueValue(self):
        if not self.edge_queue:
            return INF
        else:
            return self.edge_queue[0][0]

    def prune_edge(self, c_best):
        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point)

    def prune(self, c_best):
        self.samples = [point for point in self.samples if self.get_f_score(point) < c_best ]
        self.prune_edge(c_best)
        vertices_temp = []
        for point in self.vertices:
            if self.get_f_score(point) <= c_best:
                if self.get_g_score(point)==INF:
                    self.samples.append(point)
                else:
                    vertices_temp.append(point)
        self.vertices = vertices_temp

    def expand_vertex(self, point):
        # get the nearest value in vertex for every one in samples where difference is less than the radius
        neigbors_sample = []
        for sample in self.samples:
            if self.distance(point,sample) <= self.r:
                neigbors_sample.append(sample)

        # add an edge to the edge queue is the path might improve the solution
        for neighbor in neigbors_sample:
            estimated_f_score = self.heuristic_cost(self.start, point) + \
                self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
            if estimated_f_score < self.g_scores[self.goal]:
                heapq.heappush(self.edge_queue,(self.get_edge_value((point,neighbor)),(point,neighbor)))

        # add the vertex to the edge queue
        if point not in self.old_vertices:
            neigbors_vertex = []
            for ver in self.vertices:
                if self.distance(point,ver) <= self.r:
                    neigbors_vertex.append(ver)
            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = self.heuristic_cost(self.start, point) + \
                        self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
                    if estimated_f_score < self.g_scores[self.goal]:
                        estimated_g_score = self.get_g_score(point) + self.heuristic_cost(point, neighbor)
                        if estimated_g_score < self.get_g_score(neighbor):
                            heapq.heappush(self.edge_queue,(self.get_edge_value((point,neighbor)),(point,neighbor)))

    def get_best_path(self):
        path = []
        if self.g_scores[self.goal] != INF:
            path.append(self.goal)
            point = self.goal
            while point != self.start:
                point = self.edges[point]
                path.append(point)
            path.reverse()
        return path

    def path_length_calculate(self, path):
        path_length=0
        for i in range(len(path)-1):
            path_length += self.distance(path[i], path[i+1])
        return path_length

    def plot_function(self, path):
        env_plot = self.draw.plot_environment()
        self.draw.plot_many_points(env_plot, self.samples)
        self.draw.plot_many_points(env_plot, [self.start], pointSize=6, colorselect="green")
        self.draw.plot_many_points(env_plot, [self.goal], pointSize=6, colorselect="red")
        self.draw.plot_many_edges(env_plot, list(self.edges.items()))
        self.draw.plot_ellipse(env_plot, self.g_scores[self.goal], self.start, self.goal, color="black")
        self.draw.plot_path(env_plot, path)  # self.get_best_path()
        # plt.show()

    def plan(self, pathLengthLimit):
        radius_constant = self.setup_planning()
        path = []
        for i in range(self.maxIter):
            if not self.vertex_queue and not self.edge_queue:
                c_best = self.g_scores[self.goal]
                path = self.get_best_path()
                self.prune(c_best)
                self.samples.extend(self.informed_sample(c_best, self.batch_size))

                self.old_vertices = set(self.vertices)
                self.vertex_queue = [(self.get_point_value(point),point) for point in self.vertices]
                heapq.heapify(self.vertex_queue)    # change to op priority queue
                q = len(self.vertices)+len(self.samples)
                self.r = radius_constant * ((math.log(q) / q) ** (1.0/self.dimension))

            while self.bestVertexQueueValue() <= self.bestEdgeQueueValue():
                _, point = heapq.heappop(self.vertex_queue)
                self.expand_vertex(point)

            best_edge_value, bestEdge = heapq.heappop(self.edge_queue)

            # Check if this can improve the current solution
            if best_edge_value < self.g_scores[self.goal]:
                actual_cost_of_edge = self.actual_edge_cost(bestEdge[0], bestEdge[1])
                actual_f_edge = self.heuristic_cost(self.start, bestEdge[0]) + actual_cost_of_edge + self.heuristic_cost(bestEdge[1], self.goal)
                if actual_f_edge < self.g_scores[self.goal]:
                    actual_g_score_of_point = self.get_g_score(bestEdge[0]) + actual_cost_of_edge
                    if actual_g_score_of_point < self.get_g_score(bestEdge[1]):
                        self.g_scores[bestEdge[1]] = actual_g_score_of_point
                        self.edges[bestEdge[1]] = bestEdge[0]
                        if bestEdge[1] not in self.vertices:
                            self.samples.remove(bestEdge[1])
                            self.vertices.append(bestEdge[1])
                            heapq.heappush(self.vertex_queue,(self.get_point_value(bestEdge[1]),bestEdge[1]))

                        self.edge_queue = [ item for item in self.edge_queue if item[1][1]!=bestEdge[1] or \
                                            self.get_g_score(item[1][0]) + self.heuristic_cost(item[1][0],item[1][1])<self.get_g_score(item[1][0]) ]
                        heapq.heapify(self.edge_queue)      # Rebuild the priority queue because it will be destroyed after the element is removed

            else:
                self.vertex_queue = []
                self.edge_queue = []
                print("Step:", i, "Path Length:", self.g_scores[self.goal], "Planer: BIT*")
            if self.plot_planning_process and i!=0 and ( i%5000==0 or i==(self.maxIter-1)):
                self.plot_function(path)
                plt.show()
            if self.g_scores[self.goal] < pathLengthLimit:
                break
        if self.plot_planning_process:
            self.plot_function(path)
            plt.show()


if __name__ == '__main__':
    bounds = (-120, -120, 120, 120)
    maxIter = 100000

    # RRT_circle_enviroment
    # best path length: 196
    pathLengthLimit = 200
    environment = Environment(None,bounds)
    environment.add_obstacles([Point(0,0).buffer(40)])
    start = (0, 90)
    goal = (0,-90)


    # RRT_clutter_enviroment
    # best path length: 185.5
    # pathLengthLimit = 186
    # environment = Environment(None,bounds)
    # obstacleSet=[Polygon([(2,60), (2,70), (50,70), (50,60)]),
    #             Polygon([(-60,24), (-60,38), (10,38), (10,24)]),
    #             Polygon([(-5,-10), (-5,2), (60,2), (60,-10)]),
    #             Polygon([(-60,-40), (-60,-29), (0,-29), (0,-40)]),
    #             Polygon([(-10,-70), (-10,-60), (40,-60), (40,-70)])]
    # environment.add_obstacles(obstacleSet)
    # start = (0, 85)
    # goal = (-5,-85)

    # Set seeds
    seed = 2012
    np.random.seed(seed)

    plot_planning_process = True
    cur_time = time.time()
    bitStar = BITStar(environment,start,goal,bounds,maxIter=maxIter,plot_flag=plot_planning_process)
    time_and_cost_set = bitStar.plan(pathLengthLimit)
    print("The time is:", time.time()-cur_time)

