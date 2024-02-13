import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry
from scipy.interpolate import CubicSpline


# ### Utils class
# ### Used to draw the map and obstacles
# class Utils:

#     def drawMap(self, obs, curr, dest):
#         fig = plt.figure()
#         currentAxis = plt.gca()
#         for ob in obs:
#             circle = Circle((ob.x, ob.y), ob.r, alpha=0.4)
#             currentAxis.add_patch(circle)

#         plt.scatter(curr[0], curr[1], s=200, c="green")
#         plt.scatter(dest[0], dest[1], s=200, c="green")
#         fig.canvas.draw()


### Dijkstra's shortest path algorithm


class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]
        from_node_edges[to_node] = edge


def min_dist(q, dist):
    """
    Returns the node with the smallest distance in q.
    Implemented to keep the main algorithm clean.
    """
    min_node = None
    for node in q:
        if min_node == None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node

    return min_node


INFINITY = float("Infinity")


def dijkstra(graph, source):
    q = set()
    dist = {}
    prev = {}

    for v in graph.nodes:  # initialization
        dist[v] = INFINITY  # unknown distance from source to v
        prev[v] = INFINITY  # previous node in optimal path from source
        q.add(v)  # all nodes initially in q (unvisited nodes)

    # distance from source to source
    dist[source] = 0

    while q:
        # node with the least distance selected first
        u = min_dist(q, dist)

        q.remove(u)

        try:
            if u in graph.edges:
                for _, v in graph.edges[u].items():
                    alt = dist[u] + v.length
                    if alt < dist[v.to_node]:
                        # a shorter path to v has been found
                        dist[v.to_node] = alt
                        prev[v.to_node] = u
        except:
            pass

    return dist, prev


def to_array(prev, from_node):
    """Creates an ordered list of labels as a route."""
    previous_node = prev[from_node]
    route = [from_node]

    while previous_node != INFINITY:
        route.append(previous_node)
        temp = previous_node
        previous_node = prev[temp]

    route.reverse()
    return route


### Obstacle class


class Obstacle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        # self.calcFullCord()

    def printFullCords(self):
        print(self.x, self.y, self.r)


### PRM Controller class


class PRMController:
    def __init__(
        self, numOfRandomCoordinates, allObs, current, destination, fieldHalfSize
    ):
        self.numOfCoords = numOfRandomCoordinates
        self.coordsList = np.array([])
        self.allObs = [Obstacle(obs[0], obs[1], obs[2]) for obs in allObs]
        self.current = np.array(current)
        self.destination = np.array(destination)
        self.fieldHalfSize = fieldHalfSize
        self.graph = Graph()
        # self.utils = Utils()
        self.solutionFound = False
        # Set to true to visualize the PRM
        self.visualize = True

    def runPRM(self, initialRandomSeed):
        seed = initialRandomSeed
        # Keep resampling if no solution found
        while not self.solutionFound:
            print("Trying with random seed {}".format(seed))
            np.random.seed(seed)

            # Generate n random samples called milestones
            self.genCoords()

            # Check if milestones are collision free
            self.checkIfCollisonFree()

            # Link each milestone to k nearest neighbours.
            # Retain collision free links as local paths.
            self.findNearestNeighbour()

            # Search for shortest path from start to end node - Using Dijksta's shortest path alg
            points = self.shortestPath()

            seed = np.random.randint(1, 100000)
            self.coordsList = np.array([])
            self.graph = Graph()
        return points

    def genCoords(self):
        self.coordsList = np.random.randint(
            -self.fieldHalfSize, self.fieldHalfSize, size=(self.numOfCoords, 2)
        )
        # Adding begin and end points
        self.current = self.current.reshape(1, 2)
        self.destination = self.destination.reshape(1, 2)
        self.coordsList = np.concatenate(
            (self.coordsList, self.current, self.destination), axis=0
        )

    def checkIfCollisonFree(self):
        collision = False
        self.collisionFreePoints = np.array([])
        for point in self.coordsList:
            collision = self.checkPointCollision(point)
            if not collision:
                if self.collisionFreePoints.size == 0:
                    self.collisionFreePoints = point
                else:
                    self.collisionFreePoints = np.vstack(
                        [self.collisionFreePoints, point]
                    )
        self.plotPoints(self.collisionFreePoints)

    def findNearestNeighbour(self, k=5):
        X = self.collisionFreePoints
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        self.collisionFreePaths = np.empty((1, 2), int)

        for i, p in enumerate(X):
            # Ignoring nearest neighbour - nearest neighbour is the point itself
            for j, neighbour in enumerate(X[indices[i][1:]]):
                start_line = p
                end_line = neighbour
                if not self.checkPointCollision(
                    start_line
                ) and not self.checkPointCollision(end_line):
                    if not self.checkLineCollision(start_line, end_line):
                        self.collisionFreePaths = np.concatenate(
                            (
                                self.collisionFreePaths,
                                p.reshape(1, 2),
                                neighbour.reshape(1, 2),
                            ),
                            axis=0,
                        )

                        a = str(self.findNodeIndex(p))
                        b = str(self.findNodeIndex(neighbour))
                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[i, j + 1])
                        x = [p[0], neighbour[0]]
                        y = [p[1], neighbour[1]]

    def shortestPath(self):
        self.startNode = str(self.findNodeIndex(self.current))
        self.endNode = str(self.findNodeIndex(self.destination))

        dist, prev = dijkstra(self.graph, self.startNode)

        pathToEnd = to_array(prev, self.endNode)

        if len(pathToEnd) > 1:
            self.solutionFound = True
        else:
            return

        # X and Y coordinates of the full path
        pointsToDisplay = [(self.findPointsFromNode(path)) for path in pathToEnd]

        for i, node in enumerate(pointsToDisplay):
            collision = False
            while i + 2 < len(pointsToDisplay) and collision == False:
                next_node = (
                    pointsToDisplay[i + 2] if i + 1 < len(pointsToDisplay) else None
                )
                if next_node is not None:
                    line = shapely.geometry.LineString([node, next_node])
                    for obs in self.allObs:
                        obstacleShape = shapely.geometry.Point(obs.x, obs.y).buffer(
                            obs.r * 2
                        )
                        if line.intersects(obstacleShape):
                            collision = True
                            break
                    if not collision:
                        pointsToDisplay.pop(i + 1)
                    else:
                        collision = True
                else:
                    break

        return pointsToDisplay
        # Plotting shorest path

        if self.visualize:
            x = [int(item[0]) for item in pointsToDisplay]
            y = [int(item[1]) for item in pointsToDisplay]
            # Perform cubic spline interpolation
            t = np.arange(len(x))
            cs = CubicSpline(t, x)
            smooth_x = cs(np.linspace(0, len(x) - 1, 10))

            cs = CubicSpline(t, y)
            smooth_y = cs(np.linspace(0, len(y) - 1, 10))

            ## Smooth x and y are the coordinates of the shortest path with cubic spline interpolation

            # plt.plot(x, y, c="red", linewidth=3.5)

            pointsToEnd = [str(self.findPointsFromNode(path)) for path in pathToEnd]
            print("****Output****")

            print(
                "The quickest path from {} to {} is: \n {} \n with a distance of {}".format(
                    self.collisionFreePoints[int(self.startNode)],
                    self.collisionFreePoints[int(self.endNode)],
                    " \n ".join(pointsToEnd),
                    str(dist[self.endNode]),
                )
            )

    def checkLineCollision(self, start_line, end_line, margin=4):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.allObs:
            obstacleShape = shapely.geometry.Point(obs.x, obs.y).buffer(obs.r * margin)
            collision = line.intersects(obstacleShape)
        if collision:
            return True
        return False

    def findNodeIndex(self, p):
        return np.where((self.collisionFreePoints == p).all(axis=1))[0][0]

    def findPointsFromNode(self, n):
        return self.collisionFreePoints[int(n)]

    def plotPoints(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]

    def checkCollision(self, obs, point):
        p_x = point[0]
        p_y = point[1]
        if np.linalg.norm([p_x - obs.x, p_y - obs.y]) <= obs.r * 2:
            return True
        else:
            return False

    def checkPointCollision(self, point):
        for obs in self.allObs:
            collision = self.checkCollision(obs, point)
            if collision:
                return True
        return False
