import open3d as o3d
import numpy as np
import cv2
import os
import yaml
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
from scipy.optimize import differential_evolution, basinhopping, brute, shgo, dual_annealing
from shapely.geometry import MultiPoint, Point, Polygon, LineString
from scipy.spatial import ConvexHull
import time

CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture')

VALID_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

#Known data: polygons deminiting rooms and overall walls perimeter

# Dataset Replica
polygons = [Polygon([(-3.3, -0.52),(-1.2, -0.23),(-1.5, 2.1),(1.5, 2.6),(1.3, 4.4),(4.4, 4.8),(4.5, 3.1),(3.7, 2.9),(3.9, 0.93),(2.3, 0.72),(2.7, -1.4),(1.5, -1.5),(2.4, -9.2),(-2.1, -9.8)]),Polygon([(2.3, 0.58),(3.9, 1.1),(4.2, -0.59),(2.6, -0.86)]),Polygon([(2.6, -0.91),(5, -0.6),(5.4, -4.1),(2, -4.6),(1.6, -1.6),(2.7, -1.4)]),Polygon([(7.2, 2.1),(11, 2.7),(11, -0.7),(10, -0.81),(10, -1.7),(7.8, -2)]),Polygon([(5.5, -2.2),(8.8, -1.9),(9.3, -5.9),(6, -6.5)]),Polygon([(8.3, -6.2),(12, -5.7),(12, -9),(8.7, -9.5)]),Polygon([(10, -0.91),(12, -0.72),(12, -1.5),(14, -1.2),(14, -2.2),(10, -2.7)]),Polygon([(11, -2.8),(14, -2.2),(14, -3.2),(12, -3.4),(12, -4.1),(11, -4.4)])]
paredes = Polygon([(1.9, -4.5),(2.5, -9.2),(-2.1, -9.8),(-3.2, -0.54),(-1.2, -0.23),(-1.5, 2.2),(1.6, 2.6),(1.4, 4.4),(4.4, 4.7),(4.6, 3.1),(3.7, 2.9),(3.9, 0.93),(2.5, 0.73),(2.5, -0.069),(2.5, 0.74),(3.9, 0.94),(4.2, -0.57),(2.6, -0.81),(2.7, -1.4),(1.6, -1.6),(1.8, -3.4),(1.6, -1.6),(2.7, -1.4),(2.7, -0.79),(5, -0.5),(5.5, -4.1),(5.7, -3.9),(5.5, -2.3),(8.7, -2.0),(9.1, -4.9),(8.7, -1.9),(7.7, -1.9),(7.2, 2.2),(11, 2.7),(11, -0.69),(10, -0.84),(10, -1.6),(10, -1),(12, -0.72),(12, -1.5),(14, -1.3),(14, -2.2),(10, -2.7),(11, -3.6),(11, -2.7),(14, -2.2),(14, -3.2),(12, -3.3),(12, -4.1),(11, -4.4),(11, -5.7),(10, -5.9),(12, -5.7),(12, -9),(8.7, -9.5),(8.1, -6.2),(9.3, -5.9),(6, -6.5),(5.7, -4.1),(2, -4.6)])
borde_paredes = paredes.boundary

def select_points_by_colors(point_cloud, colors):
    colors = [np.array(color) for color in colors]
    point_colors = np.asarray(point_cloud.colors)
    selected_indices = []
    
    for color in colors:
        color_indices = np.where(np.all(point_colors == color, axis=1))[0]
        selected_indices.extend(color_indices)

    selected_point_cloud = point_cloud.select_by_index(selected_indices)
    
    return selected_point_cloud

def get_color_for_class(class_label):
    global SCANNET_COLOR_MAP
    class_index = CLASS_LABELS.index(class_label)
    class_id = VALID_CLASS_IDS[class_index]
    color = SCANNET_COLOR_MAP.get(class_id, None)
    color = [value / 255.0 for value in color]
    return color

def objective_function(pose):
    x, y, t = pose

    paredes.contains(Point(x,y))

    if paredes.contains(Point(x, y)):

        inicio = time.time()

        pcd_objective = o3d.geometry.PointCloud(nubeLocal)

        #1: translate and rotate local point cloud
        pcd_objective.translate([x,y,0])
        theta = t * np.pi / 180
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        pcd_objective.rotate(R_z, center=(x, y, 0))

        #print('Transformar nube: ', time.time()-inicio)
        
        puntos_hull = np.asarray(pcd_objective.points)
        puntos_hull = puntos_hull[:,:2]
        if np.isnan(puntos_hull).any():
            return float('inf')
        else:
            inicio = time.time()
            hull = ConvexHull(puntos_hull)
            hull_points = puntos_hull[hull.vertices]
            polygon_local = Polygon(hull_points)
            centroide = np.mean(hull_points, axis=0)
            line = LineString([centroide, (x, y)])

            borde_paredes.intersects(line)

            #print('condicion geometrica 1: ', inicio-time.time())

            if borde_paredes.intersects(line):
                return float('inf')
            else:
                #Calculate convex hull of the local cloud points and check if they coincide in the same room
                inicio = time.time()
                ious=[]
                for polygon in polygons:
                    if polygon.contains(polygon_local):
                        ious.append(1.0)
                    elif polygon.intersects(polygon_local):
                        interseccion = polygon.intersection(polygon_local)
                        ious.append(interseccion.area/polygon_local.area)
                    else:
                        ious.append(0.0)

                #print('Condicion geometrica 2', inicio-time.time())

                if any(valor>0.9 for valor in ious):
                    habIdx = np.where(np.array(ious) > 0.9)
                    habPoly = polygons[habIdx[0][0]]

                    #2: apply gaussian_kde for each kind of object and compare it against global gaussian with KL
                    mediaKL=[]
                    for i, objeto in enumerate(objetosLocal):
                        colorObjeto = [get_color_for_class(objeto)]

                        #Heatmap global
                        inicio = time.time()
                        nubeObjetoGlobal = select_points_by_colors(nubeGlobal, colorObjeto)
                        pointsGlobal = np.asarray(nubeObjetoGlobal.points)
                        pointsGlobal[:, 2] = 0
                        nubeObjetoGlobal.points = o3d.utility.Vector3dVector(pointsGlobal)
                        nubeObjetoGlobal = nubeObjetoGlobal.voxel_down_sample(voxel_size=0.05)
                        obj_pointsGlobal = np.asarray(nubeObjetoGlobal.points)
                        obj_pointsGlobal = obj_pointsGlobal[:, :2]

                        obj_pointsGlobal = [point for point in obj_pointsGlobal if habPoly.contains(Point(point))]

                        if not obj_pointsGlobal: #NO RELEVANT OBJECTS IN THE ROOM
                            return float('inf')
                        else:
                            x_objGlobal = np.array([punto[0] for punto in obj_pointsGlobal])
                            #print(x_objGlobal)
                            y_objGlobal = np.array([punto[1] for punto in obj_pointsGlobal])
                            kdeGlobal = gaussian_kde(np.vstack([x_objGlobal, y_objGlobal]))
                            #print('Heatmap gloabl: ', time.time() - inicio)

                            inicio = time.time()

                            #heatmap local
                            nubeObjeto = select_points_by_colors(pcd_objective, colorObjeto)
                            points = np.asarray(nubeObjeto.points)
                            points[:, 2] = 0
                            nubeObjeto.points = o3d.utility.Vector3dVector(points)
                            nubeObjeto = nubeObjeto.voxel_down_sample(voxel_size=0.05)
                            obj_points = np.asarray(nubeObjeto.points)
                            x_obj = obj_points[:,0]
                            y_obj = obj_points[:,1]
                            kde = gaussian_kde(np.vstack([x_obj, y_obj]))
                            #print('Heatmap local: ', time.time()-inicio)

                            inicio = time.time()

                            # Local and global heatmap comparison
                            xmin = min(x_objGlobal.min(), x_obj.min())
                            xmax = max(x_objGlobal.max(), x_obj.max())
                            ymin = min(y_objGlobal.min(), y_obj.min())
                            ymax = max(y_objGlobal.max(), y_obj.max())

                            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                            positions = np.vstack([xx.ravel(), yy.ravel()])

                            pdf1 = kdeGlobal(positions).reshape(xx.shape)
                            pdf2 = kde(positions).reshape(xx.shape)

                            eps = 1e-10
                            pdf1 = pdf1 + eps
                            pdf2 = pdf2 + eps

                            kl_div = np.sum(pdf2 * np.log(pdf2 / pdf1))

                            #print('Comparacion: ', time.time()-inicio)

                            mediaKL.append(kl_div)

                    return np.mean(mediaKL)
                else:
                    return float('inf')
    else:
        return float('inf')

print('SEMANTIC LOCALIZER')
print('Loading data...')

start=time.time()

nube_casa = o3d.io.read_point_cloud('planoColor2.ply') # Loading 2d global map
nubeGlobal = o3d.io.read_point_cloud('completaColor2.ply') # Loading 3d global map
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

# --------------------------------------------------------- Load only the local 3D map
objetosLocal = ['cabinet','toilet']
nubeLocal = o3d.io.read_point_cloud('bajo.ply')

print('Ready! Time: ',time.time()-start)

print('Optimizing...')

start = time.time()
# --------------------------------------------------------- Differential Evolution
puntosLibre = np.asarray(nube_casa.points)
min_vals = puntosLibre.min(axis=0)
max_vals = puntosLibre.max(axis=0)

bounds = [(min_vals[0], max_vals[0]), (min_vals[1], max_vals[1]), (0, 359)]

result = differential_evolution(objective_function, bounds, maxiter=20, popsize=200)

best_solution = result.x
best_fitness = result.fun

print('Ready! Total time: ',time.time()-start)
print(best_solution)
print(best_fitness)

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2, resolution=50)
mesh_sphere.translate([best_solution[0],best_solution[1],0])

nubeLocal.translate([best_solution[0],best_solution[1],0])
theta = best_solution[2] * np.pi / 180
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])
nubeLocal.rotate(R_z, center=(best_solution[0],best_solution[1],0))

o3d.visualization.draw_geometries([nubeLocal,nubeGlobal,nube_casa,mesh,mesh_sphere])

obj_points = np.asarray(nubeLocal.points)
x_obj = obj_points[:,0]
y_obj = obj_points[:,1]

puntos_hull = np.asarray(nubeLocal.points)
puntos_hull = puntos_hull[:,:2]
hull = ConvexHull(puntos_hull)
hull_points = puntos_hull[hull.vertices]
polygon_local = Polygon(hull_points)

centroide = np.mean(hull_points, axis=0)
line = LineString([centroide, (best_solution[0],best_solution[1])])
print('Intersects with wall:')
print(paredes.intersects(line))

print('Overlap percentage:')
ious=[]
for polygon in polygons:
    if polygon.contains(polygon_local):
        ious.append(1.0)
    elif polygon.intersects(polygon_local):
        interseccion = polygon.intersection(polygon_local)
        ious.append(interseccion.area/polygon_local.area)
    else:
        ious.append(0.0)
print(ious)

fig1, ax1 = plt.subplots()
ax1.scatter(x_obj, y_obj, c='green', label='Local points')
xp,yp = polygon_local.exterior.xy
ax1.plot(xp, yp, color='blue')
xp2,yp2 = paredes.exterior.xy
ax1.plot(xp2, yp2, color='green')
xp3,yp3 = line.xy
ax1.plot(xp3, yp3, color='red')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
plt.show()
