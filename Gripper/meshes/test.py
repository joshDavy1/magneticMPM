import open3d as o3d
import numpy as np

print("Testing mesh in open3d ...")
mesh = o3d.io.read_triangle_mesh("test.stl")
mesh = mesh.scale(0.1)
print(mesh)
