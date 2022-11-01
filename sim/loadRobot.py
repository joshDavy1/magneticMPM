import numpy as np
import yaml
import trimesh
import os

class Robot:
    def __init__(self, robotFile, ppm3, scale=1):
        self.robotFile = robotFile
        self.path = os.path.dirname(robotFile)

        with open(self.robotFile, 'r') as file:
            self.robotDict = yaml.safe_load(file)
        
        self.name = self.robotDict["name"]
        self.n_segments = self.robotDict["n_segments"]
        self.remenance = self.robotDict["remenance"]

        self.meshes = []
        self.magnetisations = []
        self.centers = []
        self.fixedSegments = []
        self.density = []
        self.bulk_modulus = []


        scaling_matrix = np.eye(4) * scale
        scaling_matrix[3, 3] = 1

        for i in range(self.n_segments):
            mesh = self.robotDict["segments"][i]["mesh"]
            self.meshes.append(trimesh.load(self.path+"/meshes/"+mesh))
            self.meshes[i].apply_transform(scaling_matrix)
            mag = np.array(self.robotDict["segments"][i]["magnetisation_vector"])
            self.magnetisations.append(mag)
            self.centers.append(self.meshes[i].center_mass)
            self.fixedSegments.append(self.robotDict["segments"][i]["fixed"])
            self.density.append(self.robotDict["segments"][i]["density"])
            self.bulk_modulus.append(self.robotDict["segments"][i]["bulk_modulus"])

        self.particles = []
        self.n = 0
        self.__createParticleRepresentation(ppm3)

    def __createParticleRepresentation(self, ppmm3, samplePoints=1000):
        for i in range(self.n_segments):
            print("Segment:", i)
            mesh = self.meshes[i]
            V = mesh.volume
            n_particles = int(ppmm3*V)
            #trimesh.sample.volume_mesh uses rejection sampling therefore will not return
            #the requested points. Therefore sampling is repeated until correct number of points
            # are gathered. Points are randomised
            pointsGathered = 0
            points = np.array([])
            while pointsGathered < n_particles:
                print(pointsGathered, "/", n_particles)
                p = trimesh.sample.volume_mesh(mesh, samplePoints)
                if pointsGathered + p.shape[0] > n_particles:
                    p = p[0:(n_particles - p.shape[0] - pointsGathered)]
                np.random.shuffle(p)
                pointsGathered += p.shape[0]
                if points.size == 0:
                    points = p
                else:
                    points = np.vstack([points, p])
            self.particles.append(points)
            self.n += points.shape[0]
