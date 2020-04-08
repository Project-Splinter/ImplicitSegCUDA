# import pptk
# import numpy as np
# P = np.random.rand(100,3)
# v = pptk.viewer(P)

import trimesh

mesh = trimesh.creation.icosphere()
mesh.visual.face_colors = [200, 200, 250, 100]


import vtkplotter

n = mesh.vertices.shape[0]
#Assign a color based on a scalar and a color map
pc2 = vtkplotter.Points(mesh.vertices, r=10)
# pc2.pointColors(mesh.vertices[:,2], cmap='viridis')

# Draw result on N=2 sync'd renderers, white background
vtkplotter.show([(mesh, pc2)], N=1, bg="white", axes=1)