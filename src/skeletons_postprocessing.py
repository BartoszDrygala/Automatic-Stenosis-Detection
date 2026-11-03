import numpy as np
import cv2



#***************************************************************
# MAIN PIPELINE:

#***************************************************************

#Lookin' for neighbours here
def get_neighbours(x:int, y:int, skeleton:np.ndarray):
    """
    Finding all neighbours of given point 

    Parameters
    ----------
    x : int
        x position of the point
    y : int
        y position of the point
    skeleton : np.ndarray
        Vessel skeleton

    Returns
    -------
    neighbours : list
        List of all neighbouring points
    """
    neighbours = []

    h, w = skeleton.shape

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue

            nx = x + dx
            ny = y + dy

            # correct boundary check
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue

            if skeleton[ny, nx]:
                neighbours.append((nx, ny))

    return neighbours

#Detectin' nodes, so points that are at the end or are bifurcations(when one vessels magicaly changes to two)
def detecting_nodes(skeleton:np.ndarray):
    """
    Detect all points that are either endpoints or bifurcation points

    Parameters
    ----------
    skeleton : np.ndarray
        Vessel skeleton
    
    Returns
    -------
    nodes : set
        A set of all nodal points
    """
    nodes = set()

    h, w = skeleton.shape

    for y in range(h):
        for x in range(w):
            if skeleton[y, x] == 0:
                continue

            neighbours = get_neighbours(x, y, skeleton)

            if len(neighbours) != 2:
                nodes.add((x,y))

    return nodes

# Edges, so actually it is just the vessels between the nodes, so we can separate them
def edges(start, next, skeleton, nodes):
    path = [start, next]

    prev = start
    current = next

    while True:
        neighbours = get_neighbours(*current, skeleton)

        #remove the previous pixel
        neighbours = [p for p in neighbours if p != prev]

        if len(neighbours) == 0:
            break

        next_pix = neighbours[0]

        #Stop the loop if the next pixel is the node
        if next_pix in nodes:
            path.append(next_pix)
            break

        path.append(next_pix)
        prev, current = current, next_pix
    return path


#remove bifurcations in next thingy
def remove_bifurcation_points(path, skeleton):
    clean_path = []

    for (x, y) in path:
        neighbours = get_neighbours(x, y, skeleton)

        # keep only "normal vessel points"
        if len(neighbours) == 2:
            clean_path.append((x, y))

    return clean_path


#Building paths, so now we have a fully connected thingy
def get_paths(skeleton):
    nodes = detecting_nodes(skeleton)
    visited = set()
    paths = []

    for node in nodes:
        neighbours = get_neighbours(*node, skeleton)

        for neighbour in neighbours:
            if(node, neighbour) in visited:
                continue

            path = edges(node, neighbour, skeleton, nodes)

            path = remove_bifurcation_points(path, skeleton)

            if len(path) > 5:
                paths.append(path)

            #Mark visited so the path doesnt go back or repeat itself
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                visited.add((a, b))
                visited.add((b, a))
    return paths

#***************************************************************
# EDITING OR REMOVING BRANCHES:
# 
#***************************************************************

def prune_short_branches(skeleton, min_length=10):
    skeleton = skeleton.copy()

    h, w = skeleton.shape

    for y in range(h):
        for x in range(w):
            if skeleton[y, x] == 0:
                continue

            neighbours = get_neighbours(x, y, skeleton)

            # start from endpoints only
            if len(neighbours) == 1:
                path = [(x, y)]
                current = (x, y)
                prev = None

                while True:
                    neigh = get_neighbours(*current, skeleton)
                    neigh = [n for n in neigh if n != prev]

                    if len(neigh) != 1:
                        break

                    next_pix = neigh[0]
                    path.append(next_pix)

                    prev, current = current, next_pix

                # if short → remove
                if len(path) < min_length:
                    for px, py in path:
                        skeleton[py, px] = 0

    return skeleton

def keep_main_vessels(skeleton, mask, min_ratio=0.5):
    
    #Remove very thin vessels using distance transform.
    #Keeps only branches thick enough compared to global vessel size.
    

    skeleton = skeleton.copy()

    # distance transform = vessel radius
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    # radius values only on skeleton
    ys, xs = np.where(skeleton > 0)

    if len(xs) == 0:
        return skeleton

    radii = [dist[y, x] for y, x in zip(ys, xs)]

    global_ref = np.percentile(radii, 60)

    for y, x in zip(ys, xs):
        if dist[y, x] < min_ratio * global_ref:
            skeleton[y, x] = 0

    return skeleton

#***************************************************************
# COMBINED PIPELINE
#***************************************************************

def perform_skeleton_postprocessing(skeleton):

    prunned_skeleton = prune_short_branches(skeleton, 10)
    # If masks are made correctly than leave it commented
    #skeleton = keep_main_vessels(skeleton, cv2.imread(filepath.replace("skeletons", "masks").replace("skeleton", "mask"), cv2.IMREAD_GRAYSCALE), min_ratio=0.5)
    
    paths = get_paths(prunned_skeleton)

    return paths