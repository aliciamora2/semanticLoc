import cv2
import yaml
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

def calcular_distancia_mas_cercana(imagen, fila_central, columna_central, orientacion_grados):
    vector_direccion = np.array([np.cos(np.radians(orientacion_grados)), np.sin(np.radians(orientacion_grados))])
    
    fila_actual = fila_central
    columna_actual = columna_central

    while True:
        fil = fila_actual + vector_direccion[0]
        col = columna_actual + vector_direccion[1]
        fila_siguiente = int(round(fil))
        columna_siguiente = int(round(col))

        # Check bounds to avoid index out of range errors
        if fila_siguiente < 0 or fila_siguiente >= imagen.shape[0] or columna_siguiente < 0 or columna_siguiente >= imagen.shape[1]:
            return float('inf'), float('inf'), float('inf')

        if imagen[fila_siguiente, columna_siguiente] != 254:
            distancia = np.sqrt((fila_siguiente - fila_central)**2 + (columna_siguiente - columna_central)**2)
            return distancia, columna_siguiente, fila_siguiente

        fila_actual = fil
        columna_actual = col

def objective_function(pose):
    x, y, t = pose

    if np.isnan(x) or np.isnan(y) or np.isnan(t):
        return float('inf')

    # Ensure indices are within map boundaries
    if int(y) < 0 or int(y) >= mapaGlobal.shape[0] or int(x) < 0 or int(x) >= mapaGlobal.shape[1]:
        return float('inf')

    if mapaGlobal[int(y), int(x)] != 254:
        return float('inf')

    distanciasDE = []
    for i in range(int(t), 360):
        distDE, _, _ = calcular_distancia_mas_cercana(mapaGlobal, x, y, i)
        distanciasDE.append(distDE)
    for i in range(0, int(t)):
        distDE, _, _ = calcular_distancia_mas_cercana(mapaGlobal, x, y, i)
        distanciasDE.append(distDE)

    dist = np.sum(np.abs(np.array(distanciasDE) - np.array(distanciasLocal)))

    return dist

# CALLBACK FUNCTION: Real-time visualization called after each iteration
def visualizar_evolucion(xk, convergence=None):
    print(f"Iteration completed. Current best solution: x={xk[0]:.2f}, y={xk[1]:.2f}, t={xk[2]:.2f}")
    img_with_point = mapaGlobalColor.copy()
    
    # Draw the current best candidate in green (x is column/horizontal, y is row/vertical)
    cv2.circle(img_with_point, (int(xk[0]), int(xk[1])), 3, (0, 255, 0), -1)
    
    cv2.imshow('Global map', img_with_point)
    cv2.waitKey(1)

# Load map hyperparameters
with open('mapHouse.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)
    resolution = data['resolution']
    origin = data['origin']
xOri = -origin[0]/resolution
yOri = -origin[1]/resolution

# Load global and local maps
mapaGlobal = cv2.imread('mapHouse.pgm', cv2.IMREAD_GRAYSCALE)
mapaLocal = cv2.imread('room2.pgm', cv2.IMREAD_GRAYSCALE)

# Extract local map profile (target)
distanciasLocal = []
coordenadasLocal = []
for i in range(360):
    distancia_mas_cercana, xCercano, yCercano = calcular_distancia_mas_cercana(mapaLocal, yOri, xOri, i)
    distanciasLocal.append(distancia_mas_cercana)
    coordenadasLocal.append([xCercano, yCercano])

cv2.namedWindow('Global map')
mapaGlobalColor = cv2.cvtColor(mapaGlobal, cv2.COLOR_GRAY2BGR)

bounds = [(90, 300), (110, 280), (0, 359)]

print("Evaluating initial population...")

result = differential_evolution(
    objective_function, 
    bounds, 
    strategy='rand1bin',
    mutation=(0.7, 0.7),
    recombination=0.9,
    maxiter=20, 
    popsize=900,
    callback=visualizar_evolucion
)

# Best solution found
best_solution = result.x
best_fitness = result.fun

print("\nOptimization Finished!")
print("Best robot pose found:", best_solution)
print("Objective function value for best pose:", best_fitness)

t = best_solution[2]
coordenadasGlobal = []
distanciasGlobales = []
for i in range(int(t), 360):
    d, xC, yC = calcular_distancia_mas_cercana(mapaGlobal, best_solution[0], best_solution[1], i)
    coordenadasGlobal.append([xC, yC])
    distanciasGlobales.append(d)
for i in range(0, int(t)):
    d, xC, yC = calcular_distancia_mas_cercana(mapaGlobal, best_solution[0], best_solution[1], i)
    coordenadasGlobal.append([xC, yC])
    distanciasGlobales.append(d)

coordenadasLocal = np.array(coordenadasLocal, dtype=np.int32)

coordGlobal = coordenadasLocal.copy()
coordGlobal[:, 0] = coordGlobal[:, 0] - xOri
coordGlobal[:, 1] = coordGlobal[:, 1] - yOri

sin_theta = np.sin(np.radians(best_solution[2]))
cos_theta = np.cos(np.radians(best_solution[2]))
matriz_rotacion = np.array([[cos_theta, -sin_theta],
                            [sin_theta, cos_theta]])
coordGlobal = np.dot(coordGlobal, matriz_rotacion)

coordGlobal[:, 0] = coordGlobal[:, 0] + int(best_solution[1])
coordGlobal[:, 1] = coordGlobal[:, 1] + int(best_solution[0])

coordGlobal = np.array(coordGlobal, dtype=np.int32)
coordenadasGlobal = np.array(coordenadasGlobal, dtype=np.int32)

mapaLocal = cv2.cvtColor(mapaLocal, cv2.COLOR_GRAY2BGR)

# Draw final population if available in the result object
if hasattr(result, 'population'):
    for coord in result.population:
        x, y = coord[:2]
        cv2.circle(mapaGlobalColor, (int(x), int(y)), 2, (0, 0, 255), -1)

cv2.circle(mapaLocal, (int(xOri), int(yOri)), 3, (0, 255, 0), -1)
cv2.circle(mapaGlobalColor, (int(best_solution[0]), int(best_solution[1])), 1, (0, 255, 0), -1)

cv2.polylines(mapaLocal, coordenadasLocal.reshape((-1, 1, 2)), isClosed=True, color=(255, 0, 0), thickness=2)
cv2.polylines(mapaGlobalColor, coordenadasGlobal.reshape((-1, 1, 2)), isClosed=True, color=(255, 0, 0), thickness=2)

cv2.imshow('Global map', mapaGlobalColor)
cv2.imshow('Local Map', mapaLocal)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot comparison between local and global laser scans
x_axis = np.arange(0, 360)
plt.plot(x_axis, distanciasLocal, label='Local')
plt.plot(x_axis, distanciasGlobales, label='Global')
plt.legend()
plt.show()

