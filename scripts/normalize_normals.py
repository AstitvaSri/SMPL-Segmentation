import trimesh
import os
import numpy as np
from tqdm import tqdm
import natsort
from psbody.mesh import Mesh
import pymeshlab

src = '/home/groot/PaMIR/our_scans/our_scans_image/mesh_data/'
info = '../indices/'

meshes = natsort.natsorted(os.listdir(src))

for meshname in tqdm(meshes):
    m = trimesh.load(src+meshname+'/smpl/smpl_mesh_ordered.obj',process=False)
    head = np.load(info+'head.npy')
    torso = np.load(info+'torso.npy')
    neck = np.load(info+'neck.npy')
    hair = np.load(info+'hair.npy')
    lower_torso = np.load(info+'lower_torso.npy')
    l_foot = np.load(info+'l_foot.npy')
    r_foot = np.load(info+'r_foot.npy')
    l_arm = np.load(info+'l_arm.npy')
    r_arm = np.load(info+'r_arm.npy')

    vc = 255*np.ones((m.vertices.shape[0],4)).astype('uint8')
    vc[head] = [210,200,130,255]
    vc[torso] = [255,85,85,255]
    vc[neck] = [100,100,255,255]
    vc[hair] = [50,10,10,255]
    vc[lower_torso] = [30,85,85,255]
    vc[l_foot] = [255,255,0,255]
    vc[r_foot] = [255,170,0,255]
    vc[l_arm] = [50,200,50,255]
    vc[r_arm] = [160,70,250,255]

    vis = trimesh.visual.ColorVisuals(m,vertex_colors=vc)
    m.visual = vis
    scene = trimesh.scene.Scene()
    scene.add_geometry(m)
    # scene.show('gl')
    _ = m.export(src+meshname+'/smpl/smpl_mesh_labelled.obj')

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(src+meshname+'/smpl/smpl_mesh_labelled.obj')
    ms.normalize_vertex_normals()
    ms.normalize_face_normals()
    ms.save_current_mesh(src+meshname+'/smpl/smpl_mesh_labelled.obj')
