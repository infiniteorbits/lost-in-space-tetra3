"""
This example loads the tetra3 default database and solves for every image in the
tetra3/examples/data directory.
"""

import sys
sys.path.append('..')

from PIL import Image
from pathlib import Path
EXAMPLES_DIR = Path(__file__).parent
import numpy as np
import tetra3
import json

def handle_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar to native Python scalar
    raise TypeError(f"Type {type(obj)} not serializable")

# Create instance and load the default database, built for 30 to 10 degree field of view.
# Pass `load_database=None` to not load a database, or to load your own.
 #(load_database="t3_fov20-30_mag8.npz")

path = EXAMPLES_DIR / 'data'
impath = path / '0000005.png'

# ===== SOLVING FROM IMAGE ===== ")
t3_hip1 = tetra3.Tetra3()   # Hipparcos 1 Catalogue
t3_hip2 = tetra3.Tetra3(load_database="hipparcos_2.npz")  # Hipparcos 1 Catalogue

# Path where images are
# for impath in path.glob('*'):
#     print('Solving for image at: ' + str(impath))
with Image.open(str(impath)) as img:
    # Here you can add e.g. `fov_estimate`/`fov_max_error` to improve speed or a
    # `distortion` range to search (default assumes undistorted image). There
    # are many optional returns, e.g. `return_matches` or `return_visual`. A core
    # aspect of the solution is centroiding (detecting the stars in the image).
    # You can use `return_images` to get a second return value to check the
    # centroiding process, the key `final_centroids` is especially useful.
    solution_hip1 = t3_hip1.solve_from_image(img)
    solution_hip2 = t3_hip2.solve_from_image(img)
    centroids_t3 = tetra3.get_centroids_from_image(img)
print(f'Solution from image, Hipparcos-1: {json.dumps(solution_hip1, indent=4, default=handle_numpy)}')
print(f'Solution from image, Hipparcos-2: {json.dumps(solution_hip2, indent=4, default=handle_numpy)}')
# print('Tetra3 centroids: ' + str(centroids_t3))

# ===== SOLVING FROM CENTROIDS ===== 
centroids = [
    (22.0000, 172.0000),
    (58.0000, 1322.0000),
    (281.4778, 1809.5222),
    (363.5357, 665.4929),
    (391.5174, 1176.0152),
    (446.2828, 989.0202),
    (514.0821, 1720.3455),
    (528.6808, 476.3013),
    (539.0000, 7.0000),
    (753.0000, 749.0000),
    (847.0000, 1879.0000),
    (856.5020, 1096.7631),
    (861.2842, 874.3388),
    (872.0000, 743.0000),
    (1137.0000, 1184.0000),
    (1227.7500, 1159.9601),
    (1238.0000, 945.0000),
    (1374.6011, 1609.8778),
    (1590.2715, 1874.9944),
    (1705.8259, 880.1600),
    (1749.5196, 62.0208),
    (1833.6779, 334.9551),
    (1851.0000, 766.4679),
    (1897.2766, 1577.3298),
    (1906.4833, 908.0000),
    (2035.3622, 886.7347)
]
# Invert the tuples
# centroids_inverted = centroids_t3[:, ::-1]

# Sort by the first value of each tuple
# sorted_centroids = centroids_t3[centroids_t3[:, 0].argsort()]

# print(f"SORTED CENTROIDS: {sorted_centroids}")
# print(f"TYPE: {type(centroids_t3)}")

solution_hip1 = t3_hip1.solve_from_centroids(centroids, size=(2048, 2048))
solution_hip2 = t3_hip2.solve_from_centroids(centroids, size=(2048, 2048))

print(f'Solution from centroids, Hipparcos-1: {json.dumps(solution_hip1, indent=4, default=handle_numpy)}')
print(f'Solution from centroids, Hipparcos-2: {json.dumps(solution_hip2, indent=4, default=handle_numpy)}')

"""
We want to eventually output (for each star) - 10 stars max:
- star_centroids_idx (?)
- starIDs
- u_matchedStars_C : [x, y, z] coordinate of matched star
- u_matchedStars_GCRF : [x, y, z] coordinate of matched star in GCRF
- x_px_matched
- y_px_matched
and also:
- non_star_centroids_idx : up to 100 ids/indexes?
- number of matched stars
- arclength residuals
- w_GCRF2BODY_measured
- q_GCRF2BODY_real_time_measured
- q_fit_err_axes
"""
