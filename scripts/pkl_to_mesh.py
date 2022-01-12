import sys
import pickle
import numpy as np
import smplx
import torch
import trimesh
from copy import deepcopy
from psbody.mesh import Mesh
import cv2
import os
import natsort
from tqdm import tqdm

def show(verts = None, faces = None, colors = None):
    if torch.is_tensor(verts):
        verts = verts.detach().numpy()

    if torch.is_tensor(faces):
        faces = faces.detach().numpy()
    all_meshes = []
    if faces is not None:
        for i in range(len(verts)):
            m = trimesh.Trimesh(verts[i], faces[i])
            if colors is not None:
                m.visual.vertex_colors = colors[i]
            all_meshes.append(m)
    else:
        for i in range(len(verts)):
            m = trimesh.PointCloud(verts[i], colors[i])
            all_meshes.append(m)

    scene = trimesh.scene.Scene()
    for m in all_meshes:
        scene.add_geometry(m)
    scene.show('gl')
def get_param(path, pamir = True):
    with open(path, 'rb') as fi:
        d = pickle.load(fi)

    if pamir:
        scale = d['body_scale']
        pose = d['body_pose'][0]
        beta = d['betas'][0]
        trans = d['global_body_translation']
        pose_embedding = d['body_pose_embedding']
        return pose_embedding, scale, pose, beta, trans
    else:
        scale = 1
        # print(d.keys())
        pose = d['pose']
        beta = d['betas'][:99]
        # trans = d['trans']

        return None,  scale, pose, beta, None


if __name__ == '__main__':
    
    src = '/home/groot/PaMIR/our_scans/our_scans_image/mesh_data/'

    scans = natsort.natsorted(os.listdir(src))

    for scan in tqdm(scans):


        scan_smpl_path = src + scan + '/smpl/smpl_param.pkl'

        model_folder = '../models'
        model = smplx.create(model_folder, create_global_orient = True, create_body_pose = False, create_betas = True, model_type='smpl', gender='male', create_transl = False, create_left_hand_pose= True, create_right_hand_pose = True, create_expression = True, create_jaw_pose = True, create_leye_pose = True, create_reye_pose = True, )
        pose_embedding, scale,  pose, beta, trans = get_param(scan_smpl_path)
        go = torch.tensor(pose[:3]).unsqueeze(0)
        pose = torch.tensor(pose[3:]).float().unsqueeze(0)
        beta = torch.tensor(beta).float().unsqueeze(0)
        output = model(betas=beta,  body_pose = pose, global_orient=go, return_verts=True)
        vert = output.vertices[0]
        vert = vert.detach().numpy()

        outdir = src + scan 

        mesh = Mesh()
        vert = vert*scale
        vert += trans
        mesh.v = vert
        mesh.f = model.faces
        mesh.write_obj(outdir + '/smpl/smpl_mesh_ordered.obj')
