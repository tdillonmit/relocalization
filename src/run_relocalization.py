#!/usr/bin/env python3.9

import cv2
import open3d as o3d
import numpy as np
import copy
import yaml
import time
from collections import deque
import hashlib
import json
import re
import shutil
import zipfile
import argparse
from pathlib import Path

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import tensorflow as tf

# from voxblox pybind package
from voxblox import FastTsdfIntegrator

# custom helpers
from segmentation_helpers_runtime import *
from reconstruction_helpers_runtime import *
from ultrasound_mesh_simulator import *


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "aortascope_mapping_params.yaml"
ZENODO_RECORD_ID = "20737792"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"

ZENODO_FILES = {
    "patient_1": {"filename": "patient_1.zip", "md5": "aaa5f8deff2590f8e9f9a28fc080b715"},
    "patient_2": {"filename": "patient_2.zip", "md5": "c10132d80c3259d0db822b4995430bde"},
    "patient_3": {"filename": "patient_3.zip", "md5": "f0d59c08a7cc03047f9be74b8dddaf19"},
    "patient_4": {"filename": "patient_4.zip", "md5": "85c629cc2a37694c787402c1fc57c6c0"},
    "patient_5": {"filename": "patient_5.zip", "md5": "c243b199db5ec399ada4c6ebbf7efffb"},
    "patient_6": {"filename": "patient_6.zip", "md5": "d6c291c847f0731d30676a5185834f8c"},
    "patient_7": {"filename": "patient_7.zip", "md5": "34e7de67357f7b749297c3828ac2cb50"},
    "sheep_1": {"filename": "sheep_1.zip", "md5": "b2a34d7051b66ddbe77bf831304b1d70"},
    "sheep_2": {"filename": "sheep_2.zip", "md5": "4a48a02404c0824fdeab94356da7dd21"},
    "sheep_3": {"filename": "sheep_3.zip", "md5": "feeef5837536d8bfbc05240810f6fe68"},
}
VALID_DATASETS = list(ZENODO_FILES)

DEEPLUMEN_ZENODO_RECORD_ID = "21461780"
DEEPLUMEN_ZENODO_RECORD_URL = (
    f"https://zenodo.org/records/{DEEPLUMEN_ZENODO_RECORD_ID}"
)
DEEPLUMEN_MODEL_FILENAME = "weights.weights.h5"
DEEPLUMEN_MODEL_MD5 = "fe96176b43dc226f5ce4a0f8d7f67448"
DEFAULT_DEEPLUMEN_MODEL_PATH = (
    SCRIPT_DIR / "deeplumen_model" / DEEPLUMEN_MODEL_FILENAME
)





class PointCloudUpdater:


            

    def __init__(
        self,
        dataset_path,
        config_path=DEFAULT_CONFIG_PATH,
        model_path=None,
    ):

        self.relocalization=1

        # self.write_folder = str(
        #     self._resolve_dataset_root(Path(dataset_path))
        # )

        print("dataset_path:", dataset_path)
        self.write_folder = "/home/tdillon/datasets/" + str(dataset_path)


        self.config_path = Path(config_path).expanduser().resolve()
        self.model_path_override = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else None
        )
        # Most recent segmentation overlay, saved as a reviewer-facing preview.
        self.segmentation_preview = None

        # Defaults required before the YAML configuration is loaded.
        self.dissection_mapping = 0
        self.deeplumen_slim_on = 0
        self.deeplumen_lstm_on = 0
        self.endoanchor = 0
        self.previous_no_points = 1000

        self.median_kernel=1
        closing_kernel_size=1
        self.closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
        self.min_component_size = 10
        self.saturation_value = 0.3
        self.thickness = 25
        self.area_threshold = None


        self.wireframe_gen = WireframeGenerator()


 

        
        # INITIALIZE IN MAPPING CONFIGURATION
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Mapping configuration file not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as file:
            config_yaml = yaml.safe_load(file)

        if not isinstance(config_yaml, dict):
            raise ValueError(
                f"Mapping configuration must contain a YAML mapping: "
                f"{self.config_path}"
            )

        self.load_parameters(config_yaml)

        # Use an explicitly supplied --model path as-is. Otherwise, retain an
        # existing model_path from the YAML configuration. If that configured
        # file is absent, download the released model to:
        #     ./deeplumen/weights.weights.h5
        if self.deeplumen_on == 1 and self.model_path_override is None:
            configured_model_path = (
                Path(self.model_path).expanduser().resolve()
                if self.model_path is not None
                else None
            )

            if (
                configured_model_path is not None
                and configured_model_path.is_file()
            ):
                self.model_path = str(configured_model_path)
            else:
                if configured_model_path is not None:
                    print(
                        "Configured DeepLumen weights were not found at "
                        f"{configured_model_path}. Using the released weights."
                    )
                self.model_path = str(ensure_deeplumen_model_weights())

        self.extend = 0


        # ------- INITIALIZE DEEPLUMEN ML MODEL ------- #
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
     
        if not hasattr(self, 'model'):
            self.initialize_deeplumen_model()

        self.crop_radius = 10
        self.crop_index=60
        
        

        # ---- INITIALIZE VISUALIZER ----- #
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="3D Reconstruction",
            width=1700,
            height=3000,
            left=2200,
            top=0,
        )
        self.vis.get_render_option().mesh_show_back_face = True
        self.vis.poll_events()
        self.vis.update_renderer()
    
        # ----- INITIALIZE RECONSTRUCTIONS ------ #
        self.volumetric_far_point_cloud = o3d.geometry.PointCloud()
        self.volumetric_near_point_cloud = o3d.geometry.PointCloud()

        if self.figure_mapping != 1:
            self.vis.add_geometry(self.volumetric_near_point_cloud)
        



        # ----- INITIALIZE TRACKER FRAMES ------ #
        self.frame_scaling = 0.025
        self.baseframe=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.baseframe.scale(self.frame_scaling,center=[0,0,0])

        self.us_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.us_frame.scale(self.frame_scaling,center=[0,0,0])

        # self.tracker = o3d.geometry.TriangleMesh.create_cylinder(radius=0.001, height=0.01, resolution=40)
        self.tracker = o3d.geometry.TriangleMesh.create_cylinder(radius=0.001, height=0.005, resolution=40)
        self.tracker.compute_vertex_normals()
        self.tracker.paint_uniform_color([1,0.5,0])

        self.guidewire_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.000225, height=0.004)
        self.guidewire_cylinder.compute_vertex_normals()
        self.guidewire_cylinder.paint_uniform_color([0,1,0])
        self.previous_transform_us = np.eye(4)
        self.previous_tracker_transform = np.eye(4)

        self.vis.add_geometry(self.guidewire_cylinder)
        self.vis.add_geometry(self.tracker)
        self.vis.add_geometry(self.us_frame)
        self.vis.add_geometry(self.baseframe)

    
        # ----- INITIALIZE BOUNDING BOX ----- #
        min_bounds=np.array([0.05,-0.1,-0.1]) 
        max_bounds=np.array([0.3,0.1,0.1]) 
        self.box=get_box(min_bounds,max_bounds)
        self.vis.add_geometry(self.box)

  
        # ------- INITIALIZE IMAGING PARAMETERS ------- #

        self.minimum_thickness = 15
        start_x, end_x = 59, 840
        start_y, end_y = 10, 790
        self.new_height = end_x - start_x
        self.new_width = end_y - start_y
        self.centre_x = self.new_height // 2
        self.centre_y = self.new_width // 2

        
        # BUFFER INITIALIZATION
        self.branch_buffer_size = 10
        self.lstm_length = 5

        self.grayscale_buffer = deque(maxlen=self.lstm_length)
        self.mask_1_buffer = deque(maxlen=self.branch_buffer_size)
        self.mask_2_buffer = deque(maxlen=self.branch_buffer_size)



        H, W = 224, 224

        # helper to make N zero entries
        

        # initialize two buffers
        self.mask_2A_buffer = self.init_buffer(0, self.branch_buffer_size, (H, W), 0)
        self.mask_2B_buffer = self.init_buffer(1, self.branch_buffer_size, (H, W), 0)
        self.mask_2_buffers = deque([self.mask_2A_buffer, self.mask_2B_buffer], maxlen=2)



        self.transformed_centroids = []



        # -------- INITIALIZE SDF INTEGRATOR ------#
        self.sdf_trunc = 3.5 * self.voxel_size
        self.tsdf_volume_near_lumen = FastTsdfIntegrator(
            self.voxel_size, self.sdf_trunc
        )
        self.mesh_near_lumen = o3d.geometry.TriangleMesh()

        if self.dissection_mapping == 1:
            self.vis.add_geometry(self.mesh_near_lumen)



        if(self.dissection_mapping!=1):
            self.mesh_near_lumen_lineset = o3d.geometry.LineSet()

            if(self.figure_mapping==1):
                self.tsdf_surface_pc = o3d.geometry.PointCloud()
                self.vis.add_geometry(self.tsdf_surface_pc)
                self.simple_far_pc = o3d.geometry.PointCloud()
                self.vis.add_geometry(self.simple_far_pc)
                
            else:
                self.vis.add_geometry(self.mesh_near_lumen_lineset)
                self.vis.add_geometry(self.volumetric_far_point_cloud)

        

        

    
        self.branch_pass = 0
        
        time.sleep(2)
       


        self.dest_frame = 'target1'


        # initialize view above the phantom
        view_control_1 = self.vis.get_view_control()

        view_control_1.set_up([0,1,0])

        view_control_1.set_front([0,0,-1])

        self.view_control_1 = view_control_1


        self.refine = 0
        self.pullback = 0
        self.once = 0


    @staticmethod
    def _natural_sort_key(path):
        """Sort paths containing numeric frame indices in numeric order."""
        return [
            int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", path.name)
        ]


    @classmethod
    def _resolve_dataset_root(cls, dataset_path):
        """Return the directory that directly contains image and EM-transform folders.

        Zenodo ZIP archives may preserve several parent directories. Search the
        extracted tree recursively rather than assuming zero or one wrapper
        directory.
        """
        dataset_path = dataset_path.expanduser().resolve()

        if not dataset_path.is_dir():
            return dataset_path

        image_names = ("image_numpys", "image_npys", "grayscale_images")
        transform_names = ("EM_data", "EM", "transform_data")

        def has_replay_layout(path):
            return (
                path.is_dir()
                and any((path / name).is_dir() for name in image_names)
                and any((path / name).is_dir() for name in transform_names)
            )

        if has_replay_layout(dataset_path):
            return dataset_path

        matching_directories = [
            path
            for path in dataset_path.rglob("*")
            if path.is_dir() and has_replay_layout(path)
        ]

        if not matching_directories:
            return dataset_path

        # Prefer the shallowest match. This selects an ungated acquisition over
        # deeper gated/bin_* folders when both are present.
        matching_directories.sort(
            key=lambda path: (
                len(path.relative_to(dataset_path).parts),
                str(path),
            )
        )
        shallowest_depth = len(
            matching_directories[0].relative_to(dataset_path).parts
        )
        shallowest_matches = [
            path
            for path in matching_directories
            if len(path.relative_to(dataset_path).parts) == shallowest_depth
        ]

        if len(shallowest_matches) > 1:
            ungated_matches = [
                path for path in shallowest_matches if path.name == "ungated"
            ]
            if len(ungated_matches) == 1:
                return ungated_matches[0]

            choices = "\n  - ".join(str(path) for path in shallowest_matches)
            raise RuntimeError(
                "Multiple replay datasets were found at the same archive depth. "
                "Use --dataset-path to select one explicitly:\n  - " + choices
            )

        return shallowest_matches[0]


    @staticmethod
    def _resolve_data_folder(dataset_path, candidates):
        for name in candidates:
            folder = dataset_path / name
            if folder.is_dir():
                return folder

        expected = ", ".join(candidates)
        raise FileNotFoundError(
            f"Could not find a data folder under {dataset_path}. "
            f"Expected one of: {expected}"
        )


    def _find_calibration_path(self):
        dataset_path = Path(self.write_folder)
        candidates = (
            "calibration_parameters_ivus.yaml",
            "calibration_parameters_ivus.yml",
            "calibration_parameters_ivus",
        )
        for name in candidates:
            path = dataset_path / name
            if path.is_file():
                return path

        raise FileNotFoundError(
            f"No IVUS calibration YAML was found in {dataset_path}. "
            f"Expected one of: {', '.join(candidates)}"
        )


    def _load_dataset_calibration(self):
        calibration_path = self._find_calibration_path()
        with calibration_path.open("r", encoding="utf-8") as file:
            calibration = yaml.safe_load(file)

        if not isinstance(calibration, dict):
            raise ValueError(
                f"Calibration file must contain a YAML mapping: "
                f"{calibration_path}"
            )

        required = ('/angle', '/translation', '/radial_offset', '/oclock')
        missing = [key for key in required if key not in calibration]
        if missing:
            raise KeyError(
                f"Calibration file {calibration_path} is missing: "
                f"{', '.join(missing)}"
            )

        self.calib_yaml = calibration
        self.angle = calibration['/angle']
        self.translation = calibration['/translation']
        self.radial_offset = calibration['/radial_offset']
        self.o_clock = calibration['/oclock']

        if '/scaling' in calibration:
            self.scaling = calibration['/scaling']
            self.default_values['/scaling'] = calibration['/scaling']

        return calibration_path


    def load_replay_data(self):
        dataset_path = Path(self.write_folder)

        image_folder = self._resolve_data_folder(
            dataset_path,
            ("image_numpys", "image_npys", "grayscale_images"),
        )
        transform_folder = self._resolve_data_folder(
            dataset_path,
            ("EM_data", "EM", "transform_data"),
        )

        image_files = sorted(
            image_folder.glob("*.npy"),
            key=self._natural_sort_key,
        )
        transform_files = sorted(
            transform_folder.glob("*.npy"),
            key=self._natural_sort_key,
        )

        if not image_files:
            raise FileNotFoundError(
                f"No NumPy image files were found in {image_folder}"
            )
        if not transform_files:
            raise FileNotFoundError(
                f"No NumPy transform files were found in "
                f"{transform_folder}"
            )
        if len(image_files) != len(transform_files):
            raise ValueError(
                "The image and transform counts do not match: "
                f"{len(image_files)} images versus "
                f"{len(transform_files)} transforms."
            )

        grayscale_images = [
            np.load(path, allow_pickle=False) for path in image_files
        ]
        em_transforms = [
            np.load(path, allow_pickle=False) for path in transform_files
        ]

        for path, transform in zip(transform_files, em_transforms):
            if transform.shape != (4, 4):
                raise ValueError(
                    f"Expected a 4 x 4 transform in {path}, "
                    f"but found shape {transform.shape}."
                )

        return grayscale_images, em_transforms


    def initialize_deeplumen_model(self):
        
        if(self.deeplumen_on == 1):

            if self.model_path is None:
                raise ValueError(
                    "deeplumen_on is enabled, but no model path was "
                    "provided in the YAML configuration or with --model."
                )
            if not Path(self.model_path).is_file():
                raise FileNotFoundError(
                    f"Segmentation model weights not found: "
                    f"{self.model_path}"
                )

            model = build_mldr_drn(input_shape=(224,224,3), num_classes=3, base=64,
                        blocks_per_stage=(2,2,3,3,3), dilations=(1,2,4),
                        dropout=0.2, upsample_stride=8,
                        return_pyramid=False, name="MLDR_DRN_Large")

     
            model.load_weights(self.model_path)

            model.summary()

            model = tf.function(model, jit_compile=True)
     

            self.model = model

        


    def load_parameters(self, config_yaml):

        
        # healthy first return mapping for reference
        self.tsdf_map = config_yaml['tsdf_map']

        self.voxel_size = config_yaml['voxel_size']

        self.hybrid_seg = config_yaml['hybrid_seg']

        self.conf_threshold = config_yaml['conf_threshold']

        self.deeplumen_on = config_yaml['deeplumen_on']

        self.figure_mapping = config_yaml['figure_mapping']



        configured_model_path = config_yaml.get('model_path')
        if self.model_path_override is not None:
            self.model_path = str(self.model_path_override)
        elif configured_model_path:
            configured_model_path = Path(configured_model_path).expanduser()
            if not configured_model_path.is_absolute():
                configured_model_path = (
                    self.config_path.parent / configured_model_path
                )
            self.model_path = str(configured_model_path.resolve())
        else:
            self.model_path = None

        self.vpC_map = config_yaml['vpC_map']

   

        # Load mapping defaults used by the offline replay.
        self.default_values=load_default_values(self.write_folder + '/calibration_parameters_ivus.yaml')

        # mapping specific parameters
        self.default_values['/no_points'] = 1000




        self.threshold = self.default_values['/threshold']
        self.no_points = self.default_values['/no_points']
        self.previous_no_points = self.no_points

        self.scaling = self.default_values['/scaling']

   

        
        # Dataset-specific calibration is loaded at replay time.
        self.calib_yaml = {
            '/angle': 0.0,
            '/translation': 0.0,
            '/radial_offset': 0.0,
            '/oclock': 0.0,
            '/scaling': self.default_values['/scaling'],
        }

        self.angle = self.calib_yaml['/angle']
        self.translation = self.calib_yaml['/translation']
        self.radial_offset = self.calib_yaml['/radial_offset']
        self.o_clock = self.calib_yaml['/oclock']



    def load_registetered_ct(self):

        no_reg=0
        self.once = 0
        refine=0
       

        print("loading registration lineset")
        self.registered_ct_lineset = o3d.io.read_line_set(self.write_folder+ '' + '/final_registration.ply')
        self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/final_registration_mesh.ply')       
        ct_centroid_pc = o3d.io.read_point_cloud(self.write_folder + '/side_branch_centrelines.ply')
       


        self.registered_ct_mesh.remove_unreferenced_vertices()

        self.registered_ct_mesh_2 = copy.deepcopy(self.registered_ct_mesh)

    
        self.registered_ct_lineset.paint_uniform_color([0,0,0])

        
        self.registered_ct_mesh_2.compute_vertex_normals()


        # flip_mesh_orientation(self.registered_ct_mesh_2) 



        # Get current camera parameters
        view_control_1 = self.vis.get_view_control()

        points = np.asarray(self.registered_ct_lineset.points)  # Convert point cloud to numpy array
        self.registered_centroid = np.mean(points, axis=0) 
        view_control_1.set_lookat(self.registered_centroid)

        view_control_1.set_up([0,-1,0])

        view_control_1.set_front([0,0,1])
        
        view_control_1.set_zoom(0.25)

    

        self.tsdf_map = 0

        self.view_control_1 = view_control_1

        self.vis.add_geometry(self.registered_ct_mesh)





    def replay_function(self):
        # REPLAY GATED DATA

        
            

        high_level_path = self.write_folder[:-12]
        if(self.write_folder.endswith('bin_0')==True):
            for i in np.arange(9):
                self.write_folder = high_level_path + '/gated/bin_' + str(i)
                print("RUNNING BIN:", self.write_folder)
                self.replay_iteration()

        # REPLAY NORMAL DATA
        else:
            self.replay_iteration()
            

    def replay_iteration(self):
        try:
            calibration_path = self._load_dataset_calibration()
            print(
                "loaded calibration file for dataset:",
                calibration_path,
            )
            print("offset angle is!", self.angle)

        except (FileNotFoundError, KeyError, ValueError) as error:
            print("NO CALIBRATION FOUND")
            print(error)
            self.angle = 0
            self.translation = 0
            self.radial_offset = 0
            self.o_clock = 0

        self.extend = 1

        grayscale_images, em_transforms = self.load_replay_data()

        print("pulling from folder", self.write_folder)
        print("number of loaded images:", len(grayscale_images))

        if(self.relocalization==1):
            self.load_registetered_ct()
            TEM_C = [[1,0,0,self.translation],[0,np.cos(self.angle),-np.sin(self.angle),self.radial_offset*np.cos(self.o_clock)],[0,np.sin(self.angle),np.cos(self.angle),self.radial_offset*np.sin(self.o_clock)],[0, 0, 0, 1]]  
            TEM_C = np.asarray(TEM_C)

            self.sim = VesselUltrasoundSimulator(
            self.write_folder, self.registered_ct_mesh , image_size=224, fov_mm=self.scaling * 224 * (781/224))

        for i, (grayscale_image, TW_EM) in enumerate(
            zip(grayscale_images, em_transforms)
        ):
            # try:
            print("image number:", i)
            
            # giving it the exact mask for now for comparison

            if(self.relocalization==1):
                self.sim_mask_1, self.sim_mask_2 = self.sim.simulate_segmentation(TW_EM @ TEM_C)
                self.sim_mask_1 = cv2.flip(self.sim_mask_1, 0)
                self.sim_mask_2 = cv2.flip(self.sim_mask_2, 0)

            self.append_image_transform_pair(TW_EM, grayscale_image)
            time.sleep(0.030)

            # except:
            #     print("image skipped on replay!")


        try:
            self.vis.remove_geometry(self.us_frame)
        except:
            print("no us frame present")

        

        self.extend=0


        try:
            self.vis.remove_geometry(self.mesh_near_lumen)
        except:
            print("no near lumen present")

        try:
            self.vis.remove_geometry(self.volumetric_near_point_cloud)
        except:
            print("no volumetric near point cloud present")

        try:
            self.vis.remove_geometry(self.volumetric_far_point_cloud)
        except:
            print("no volumetric far point cloud present")


        if(self.gating==1):
            self.lightweight_reinitialize()


    

        print("saved! restarted aortascope")
        # self.vis.run()



    

    def init_buffer(self,branch_id, buffer_size, shape, branch_pass_trigger):
            H, W = shape
            zero_mask = np.zeros((H, W), dtype=np.uint8)
            return deque([[branch_id, zero_mask.copy(), np.nan, branch_pass_trigger] for _ in range(buffer_size)],
                        maxlen=buffer_size)


    def get_catheter_transform(self,TW_EM):

        roll_axis = TW_EM[:3, 0]  # This will be aligned with the cylinder's roll axis
        short_axis_1 = TW_EM[:3, 1]  # This will be aligned with the first short axis of the cylinder
        short_axis_2 = TW_EM[:3, 2]  # This will be aligned with the second short axis of the cylinder

        # Normalize the axes to ensure they are unit vectors
        roll_axis = roll_axis / np.linalg.norm(roll_axis)
        short_axis_1 = short_axis_1 / np.linalg.norm(short_axis_1)
        short_axis_2 = short_axis_2 / np.linalg.norm(short_axis_2)

        # Construct the rotation matrix using the normalized basis vectors
        rotation_matrix = np.column_stack((short_axis_2, short_axis_1, roll_axis))

        T_transformation = np.eye(4)
        T_transformation[:3,:3]= rotation_matrix
        T_transformation[:3,3] = TW_EM[:3,3]
        
        T_catheter = T_transformation

        return T_catheter

    def keep_largest_component(self,mask):
        """
        mask: binary image (0/1 or 0/255)

        returns:
            largest component mask (uint8 0/1)
        """

        mask_u8 = (mask > 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8
        )

        # label 0 = background
        if num_labels <= 1:
            return mask_u8

        # areas of components (skip background)
        areas = stats[1:, cv2.CC_STAT_AREA]

        largest_label = 1 + np.argmax(areas)

        largest_mask = (labels == largest_label).astype(np.uint8)

        return largest_mask
    

    

    
 


    def append_image_transform_pair(self, TW_EM, grayscale_image, prior_phase = None):



        original_image = grayscale_image.copy()
        
        # fetch parameters for real time mapping 
        threshold = self.threshold
        crop_index = self.crop_index
        scaling = self.scaling
        angle = self.angle
        translation = self.translation
        radial_offset = self.radial_offset
        oclock = self.o_clock


        centre_x=self.centre_x
        centre_y=self.centre_y

        

        # ------ FIRST RETURN SEGMENTATION -------- #

        final_component_data = []

        mask_1=None
        mask_2=None
       

        # first return segmentation
        

        if(self.hybrid_seg == 1): 


          
            mask_1 = np.zeros_like(grayscale_image, dtype=np.uint8)  # Create a black mask
            

            filtered_image, labels, stats=morphological_processing(grayscale_image, self.median_kernel,self.closing_kernel, self.min_component_size, threshold)
            
          
            _, _, spline_pixels = spline_first_return_segmentation(filtered_image,threshold, crop_index,self.gridlines,self.thickness, self.saturation_value, original_image, labels, stats, self.area_threshold, centre_x,centre_y, self.previous_no_points)
           

            cv2.fillPoly(mask_1, [spline_pixels], 255)  # Filled white ellipse on black mask
       

            

            mask_1_hybrid = cv2.resize(mask_1, (224, 224))


        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # print("deeplumen", self.deeplumen_on)

        if(((self.deeplumen_on == 1 or (self.deeplumen_slim_on == 1 or self.deeplumen_lstm_on == 1)) and self.dest_frame=='target1') or self.endoanchor==1):

            # print("segmenting")

            if(self.deeplumen_on == 1 or self.deeplumen_slim_on == 1 or self.endoanchor == 1):
            
             

            
                

                # # note 224,224 image for compatibility with network is hardcoded
                grayscale_image = cv2.resize(grayscale_image, (224, 224))
                image = cv2.cvtColor(grayscale_image,cv2.COLOR_GRAY2RGB)


                # #---------- SEGMENTATION --------------#


                start_time = time.time()

                

                pred, conf_class2 = deeplumen_segmentation(image, self.model)
                conf_class2 = conf_class2.numpy()
                raw_data = pred[0].numpy()



                


                
                 
                if(self.hybrid_seg == 1):
                    mask_1_hybrid_bool = mask_1_hybrid > 0
                    raw_data_updated = raw_data.copy()

                    # Clear old label-1
                    raw_data_updated[raw_data == 1] = 0

                    # Write new label-1
                    safe_mask = mask_1_hybrid_bool & (raw_data_updated != 2)
                    raw_data_updated[safe_mask] = 1
                    raw_data = raw_data_updated

                
 
                mask_1, mask_2, largest_two_masks, spline_pixels = post_process_deeplumen(raw_data, conf_class2, self.conf_threshold, self.hybrid_seg)

               
                mask_2 = self.keep_largest_component(mask_2)


 
                   

                end_time=time.time()
                diff_time=end_time-start_time
                print("segmentation time:", diff_time)


        
    
            # ---- PUBLISH IMAGE (1ms) ----- #


            mask_1_contour, _ = cv2.findContours(
                mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            mask_2_contour, _ = cv2.findContours(
                mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # visualize segmentations
            grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(grayscale_image, mask_1_contour, -1, (0, 0, 255), thickness=1)
            cv2.drawContours(grayscale_image, mask_2_contour, -1, (255, 0, 0), thickness=1)

    
            mask_1_send = cv2.resize(mask_1, (np.shape(original_image)[0], np.shape(original_image)[1]))
            mask_2_send = cv2.resize(mask_2, (np.shape(original_image)[0], np.shape(original_image)[1]))


            #FILLED TRANSPARENT MASKS

            # Create color overlay
            h, w = original_image.shape[:2]
            mask_1_send = cv2.resize(mask_1, (w, h))
            mask_2_send = cv2.resize(mask_2, (w, h))
            mask_1_bin = (mask_1_send > 0).astype(np.uint8)
            mask_2_bin = (mask_2_send > 0).astype(np.uint8)
            overlay = np.zeros_like(original_image, dtype=np.uint8)
            overlay[mask_1_bin == 1] = (0, 0, 255)
            overlay[mask_2_bin == 1] = (255, 0, 0)
            alpha = 0.38  # transparency
            blended = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)
            # no_mask = (mask_1_bin == 0) & (mask_2_bin == 0)
            original_image=blended

        # Retain the most recent successful segmentation overlay for export.
        self.segmentation_preview = original_image.copy()

        # visualize image
        cv2.imshow("observed image", original_image)
        

        # sim image for comparison
        overlay_sim = np.zeros_like(original_image, dtype=np.uint8)
        sim_mask_1_send = cv2.resize(self.sim_mask_1, (w, h))
        sim_mask_2_send = cv2.resize(self.sim_mask_2, (w, h))
        # preview = sim.render_preview(self.sim_mask_1, self.sim_mask_2)
        sim_mask_1_bin = (sim_mask_1_send > 0).astype(np.uint8)
        sim_mask_2_bin = (sim_mask_2_send > 0).astype(np.uint8)
        overlay_sim[sim_mask_1_bin == 1] = (0, 0, 255)
        overlay_sim[sim_mask_2_bin == 1] = (255, 0, 0)
        
        cv2.imshow("simulated image", overlay_sim)
        cv2.waitKey(1)

        
        
  
        scaling=self.default_values['/scaling'] 


        
        if self.vpC_map == 1:
            volumetric_three_d_points_near_lumen = get_single_point_cloud_from_mask(mask_1, scaling)
            volumetric_three_d_points_far_lumen = get_single_point_cloud_from_mask(mask_2, scaling) 
        
            

        # ---- KINEMATICS ---- #

        angle=self.calib_yaml['/angle'] 
        translation=self.calib_yaml['/translation'] 
        radial_offset=self.calib_yaml['/radial_offset'] 
        oclock=self.calib_yaml['/oclock'] 

       
        TEM_C = [[1,0,0,translation],[0,np.cos(angle),-np.sin(angle),radial_offset*np.cos(oclock)],[0,np.sin(angle),np.cos(angle),radial_offset*np.sin(oclock)],[0, 0, 0, 1]]

        
        TEM_C = np.asarray(TEM_C)

        extrinsic_matrix = TW_EM @ TEM_C


        # ---- ADD TO BUFFERS ----- #
        if(self.dest_frame == 'target2'):
            mask_1 = np.zeros_like(grayscale_image)
            mask_2 = np.zeros_like(grayscale_image)


        if(self.dest_frame == 'target1' and np.count_nonzero(mask_1)>0):
            self.mask_1_buffer.append(mask_1)
            self.mask_2_buffer.append(mask_2)

            # ---- 2D CENTROID CALCULATIONS ----- #
            

            moments = cv2.moments(mask_1)
            centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            centre_x = 224.0/2.0
            centre_y = 224.0/2.0
            centred_centroid=np.array(centroid)-[centre_x,centre_y]
            # centred_centroid=np.array(centroid)
            two_d_centroid=centred_centroid*scaling


        # post process the bspline smoothing instead - this prevents overheating
        if(self.extend==1 and self.dissection_mapping != 1):

                extrinsic_matrix = TW_EM @ TEM_C
                
                # ---- GET 3D CENTROIDS ----- #
                three_d_centroid = np.hstack((0,two_d_centroid))
                three_d_centroid=np.hstack((three_d_centroid,1)).T
                transformed_centroid = extrinsic_matrix @ three_d_centroid
                self.transformed_centroids.append(transformed_centroid)

                
        # ------- VOXBLOX TSDF MESHING -------- #
        
        if(self.extend == 1):
            if(self.tsdf_map == 1 ):
            #if(self.tsdf_map == 1 and self.refine==0):
                combined_mask = cv2.bitwise_or(mask_1, mask_2)
                combined_mask = np.uint8(combined_mask)

                

                _, three_d_points_near_lumen, _, _ = get_point_cloud_from_masks(
                    combined_mask, scaling, mask_1_contour, mask_2_contour
                )

                
                

                if(three_d_points_near_lumen is not None):
     
                    

                    update_tsdf_mesh(self.vis, self.tsdf_volume_near_lumen,self.mesh_near_lumen,three_d_points_near_lumen, extrinsic_matrix,[1,0,0], keep_largest=False)

                    
                    

                    if(self.dissection_mapping!=1 and np.shape(np.array(self.mesh_near_lumen.vertices))[0]>0):
             
                  
                        temp_lineset = self.wireframe_gen.update_from_mesh(self.mesh_near_lumen)
               
                 
                        self.mesh_near_lumen_lineset.points = temp_lineset.points
                        self.mesh_near_lumen_lineset.lines = temp_lineset.lines
                        if(self.refine==1):
                            self.mesh_near_lumen_lineset.paint_uniform_color([0.1, 0.7, 0.8])
                            self.vis.update_geometry(self.mesh_near_lumen_lineset)
                        # else:

                        # if(self.figure_mapping==1):

                        #     self.mesh_near_lumen_lineset.paint_uniform_color([0,0,1])

                        if(self.refine ==0):
                            self.vis.update_geometry(self.mesh_near_lumen_lineset)

                            if(self.figure_mapping==1):
                                self.tsdf_surface_pc.points = self.mesh_near_lumen_lineset.points
                                self.tsdf_surface_pc.paint_uniform_color([0,0,1])
                                self.vis.update_geometry(self.tsdf_surface_pc)

                
        # ------- VOLUMETRIC POINT CLOUD 3D ----- #
        if(self.vpC_map == 1):
            
            
            if(volumetric_three_d_points_near_lumen is not None):
                near_vpC_points=o3d.geometry.PointCloud()
                near_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_near_lumen)

                # downsample volumetric point cloud
                near_vpC_points = near_vpC_points.voxel_down_sample(voxel_size=0.0005)

                near_vpC_points.transform(TW_EM @ TEM_C)

               


                # if you want all the point cloud points (will be really slow)
                # if(self.extend == 1):
                if(self.extend == 1 and (self.dissection_mapping == 1 or self.pullback==0)):
                    # prevent memory issues by commenting this out
                    self.volumetric_near_point_cloud.points.extend(near_vpC_points.points)
                    pass
                else:
                   
                    self.volumetric_near_point_cloud.points = near_vpC_points.points


            

                self.volumetric_near_point_cloud.paint_uniform_color([1,0,0])


            
            
            # run this for each component -> volumetric threed points, branch pass, number of branch pixels
            for final_component in final_component_data:

                branch_pass = final_component[0]

                volumetric_three_d_points_far_lumen = final_component[2]
                branch_pixels = final_component[3]

                if(volumetric_three_d_points_far_lumen is not None):
        
                    
                
                    far_vpC_points=o3d.geometry.PointCloud()
                    far_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_far_lumen)

                    #downsample volumetric point cloud
                    far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0005)

                    if(self.pullback==1): # downsample again to prevent memory issues
                        far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0025)

                    far_vpC_points.transform(TW_EM @ TEM_C)


                    # for results evaluation
                    max_branch_pass = 255  # Set based on your application needs
 
                    normalized_pass = branch_pass / max_branch_pass  # Scale to [0, 1]
                    max_branch_pixels = 2000.0
                    normalized_branch_pixels = branch_pixels / max_branch_pixels
                    duplicated_pass_colors = np.repeat([[normalized_pass, normalized_branch_pixels, 0]], len(far_vpC_points.points), axis=0)

    
                    far_vpC_points.colors = o3d.utility.Vector3dVector(duplicated_pass_colors)

                    if(self.extend == 1):

                        self.volumetric_far_point_cloud.points.extend(far_vpC_points.points)
                        self.volumetric_far_point_cloud.colors.extend(far_vpC_points.colors)

                        

                    else:
                        self.volumetric_far_point_cloud.points = far_vpC_points.points
                        self.volumetric_far_point_cloud.colors = far_vpC_points.colors

                    if(self.figure_mapping==1):
                        # note this will mess up branch pass
                        self.simple_far_pc.points = copy.deepcopy(self.volumetric_far_point_cloud.points)
                        self.simple_far_pc.paint_uniform_color([0,0,1])
                     
                    


          
        

            # self.vis.update_geometry(self.point_cloud)
            #JUST DONT VISUALIZE IT
            self.vis.update_geometry(self.volumetric_near_point_cloud)
            self.vis.update_geometry(self.volumetric_far_point_cloud)
            if self.figure_mapping == 1:
                self.vis.update_geometry(self.simple_far_pc)
            



        # ----- FOLLOW THE PROBE ------- #


        if(self.tsdf_map ==1):
            vertices_of_interest = np.asarray(self.mesh_near_lumen.vertices)
            if(vertices_of_interest is not None):

    



                if(np.size(vertices_of_interest)>0):
                    centroid = np.mean(vertices_of_interest, axis=0) 
                else:
                    centroid = np.asarray([0,0,0])
                position_tracker = TW_EM[:3,3]
                average_point = (centroid+position_tracker)/2
                lookat = average_point

                
                if(self.once == 0):
                    up = np.array([0, -1, 0])
                    self.view_control_1.set_up(up)
                    self.view_control_1.set_front([0,0,-1])
                    self.once=1
                    self.view_control_1.set_zoom(0.5)

                
                self.view_control_1.set_lookat(lookat)

                
                

    
     
                
                
    

  


        # ------ TRACKER FRAMES ------ #
 

        

        # tracker
        T_tracker = self.get_catheter_transform(TW_EM)
        self.tracker.transform(get_transform_inverse(self.previous_tracker_transform))
        self.tracker.transform(T_tracker)
        self.previous_tracker_transform = T_tracker
        self.vis.update_geometry(self.tracker)



        self.us_frame.transform(get_transform_inverse(self.previous_transform_us))
        self.us_frame.transform(TEM_C)
        self.us_frame.transform(TW_EM)
        self.vis.update_geometry(self.us_frame)

        self.previous_transform_us = TW_EM @ TEM_C

        self.vis.poll_events()
        self.vis.update_renderer()


   

def download_and_extract_dataset(dataset_name, data_root, force_download=False):
    """Ensure a named Zenodo dataset is downloaded, verified, and extracted."""
    if dataset_name not in ZENODO_FILES:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Valid choices: "
            f"{', '.join(VALID_DATASETS)}"
        )

    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_path = data_root / dataset_name

    if _dataset_layout_exists(dataset_path) and not force_download:
        print(f"Using existing extracted dataset: {dataset_path}")
        return dataset_path

    if dataset_path.exists() and not force_download:
        raise RuntimeError(
            f"{dataset_path} exists but does not contain the expected "
            "image_numpys/EM_data dataset structure. Remove it or rerun with "
            "--force-download."
        )

    metadata = ZENODO_FILES[dataset_name]
    filename = metadata["filename"]
    expected_md5 = metadata["md5"].lower()
    archive_dir = data_root / ".downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / filename
    download_url = f"{ZENODO_RECORD_URL}/files/{filename}?download=1"

    archive_is_valid = (
        archive_path.is_file()
        and _calculate_md5(archive_path).lower() == expected_md5
    )
    if force_download or not archive_is_valid:
        if archive_path.exists():
            reason = "forced redownload" if force_download else "invalid archive"
            print(f"Removing existing archive ({reason}): {archive_path}")
            archive_path.unlink()
        print(f"Downloading {dataset_name} from Zenodo record {ZENODO_RECORD_ID}...")
        _download_with_progress(download_url, archive_path)
    else:
        print(f"Using previously downloaded archive: {archive_path}")

    actual_md5 = _calculate_md5(archive_path).lower()
    if actual_md5 != expected_md5:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum verification failed for {archive_path.name}. "
            f"Expected {expected_md5}, obtained {actual_md5}. "
            "The invalid archive was removed; rerun the command to download it again."
        )
    print(f"Verified MD5 checksum: {actual_md5}")

    extraction_path = data_root / f".{dataset_name}.extracting"
    if extraction_path.exists():
        shutil.rmtree(extraction_path)

    print(f"Extracting {archive_path.name}...")
    try:
        _safe_extract_zip(archive_path, extraction_path)
        resolved_root = PointCloudUpdater._resolve_dataset_root(extraction_path)
        if not _dataset_layout_exists(resolved_root):
            discovered_directories = sorted(
                str(path.relative_to(extraction_path))
                for path in extraction_path.rglob("*")
                if path.is_dir()
            )
            preview = "\n  - ".join(discovered_directories[:30])
            if len(discovered_directories) > 30:
                preview += "\n  - ..."
            raise RuntimeError(
                "The extracted archive does not contain a directory with both "
                "image_numpys (or image_npys/grayscale_images) and EM_data (or EM/transform_data). "
                f"Extraction root: {extraction_path}"
                + (f"\nDiscovered directories:\n  - {preview}" if preview else "")
            )

        # Only replace an existing dataset after the new archive has been
        # downloaded, verified, extracted, and validated successfully.
        if dataset_path.exists():
            if dataset_path.is_dir():
                shutil.rmtree(dataset_path)
            else:
                dataset_path.unlink()

        if resolved_root == extraction_path:
            extraction_path.replace(dataset_path)
        else:
            shutil.move(str(resolved_root), str(dataset_path))
            shutil.rmtree(extraction_path, ignore_errors=True)
    except Exception:
        shutil.rmtree(extraction_path, ignore_errors=True)
        raise

    print(f"Dataset ready: {dataset_path}")
    return dataset_path



def _download_with_progress(url, destination):
    """Download a URL to destination using only the Python standard library."""
    destination = Path(destination)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(
        url,
        headers={"User-Agent": "AortaScope-mapping/1.0 (Zenodo dataset downloader)"},
    )

    try:
        with urlopen(request, timeout=60) as response, partial_path.open("wb") as output:
            total_header = response.headers.get("Content-Length")
            total_bytes = int(total_header) if total_header else None
            downloaded = 0
            chunk_size = 1024 * 1024

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    percent = 100.0 * downloaded / total_bytes
                    print(
                        f"\rDownloading {destination.name}: "
                        f"{downloaded / 1024**2:.1f}/{total_bytes / 1024**2:.1f} MB "
                        f"({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rDownloading {destination.name}: "
                        f"{downloaded / 1024**2:.1f} MB",
                        end="",
                        flush=True,
                    )
    except (HTTPError, URLError, TimeoutError) as error:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {error}") from error
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    print()
    partial_path.replace(destination)

def _dataset_layout_exists(dataset_path):
    """Return True when a path contains a usable extracted replay dataset."""
    resolved = PointCloudUpdater._resolve_dataset_root(Path(dataset_path))
    image_names = ("image_numpys", "image_npys", "grayscale_images")
    transform_names = ("EM_data", "EM", "transform_data")
    return (
        resolved.is_dir()
        and any((resolved / name).is_dir() for name in image_names)
        and any((resolved / name).is_dir() for name in transform_names)
    )
  
def _calculate_md5(path, chunk_size=1024 * 1024):
    """Calculate the MD5 checksum of a file without loading it into memory."""
    digest = hashlib.md5()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_deeplumen_model_weights(
    destination=DEFAULT_DEEPLUMEN_MODEL_PATH,
):
    """Download and verify the released DeepLumen weights when needed."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_md5 = DEEPLUMEN_MODEL_MD5.lower()

    if destination.exists() and not destination.is_file():
        raise RuntimeError(
            "DeepLumen model destination exists but is not a file: "
            f"{destination}"
        )

    if destination.is_file():
        actual_md5 = _calculate_md5(destination).lower()

        if actual_md5 == expected_md5:
            print(f"Using existing DeepLumen model weights: {destination}")
            print(f"Verified model MD5 checksum: {actual_md5}")
            return destination

        print(
            "Removing DeepLumen model weights with an invalid checksum: "
            f"{destination}"
        )
        destination.unlink()

    download_url = (
        f"{DEEPLUMEN_ZENODO_RECORD_URL}/files/"
        f"{DEEPLUMEN_MODEL_FILENAME}?download=1"
    )

    print(
        "Downloading DeepLumen model weights from Zenodo record "
        f"{DEEPLUMEN_ZENODO_RECORD_ID}..."
    )
    _download_with_progress(download_url, destination)

    actual_md5 = _calculate_md5(destination).lower()
    if actual_md5 != expected_md5:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "Checksum verification failed for the downloaded DeepLumen model. "
            f"Expected {expected_md5}, obtained {actual_md5}. "
            "The invalid file was removed; rerun the command to download it "
            "again."
        )

    print(f"Verified model MD5 checksum: {actual_md5}")
    print(f"DeepLumen model weights ready: {destination}")
    return destination



def _safe_extract_zip(archive_path, destination):
    """Extract a ZIP while rejecting paths that escape the destination."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(destination_root)
            except ValueError as error:
                raise ValueError(
                    f"Unsafe path in ZIP archive {archive_path}: {member.filename}"
                ) from error
        archive.extractall(destination)



def _geometry_has_points(geometry):
    """Return True when an Open3D geometry contains vertices or points."""
    if hasattr(geometry, "vertices"):
        return len(geometry.vertices) > 0
    if hasattr(geometry, "points"):
        return len(geometry.points) > 0
    return False


def save_mapping_outputs(mapper, output_dir, runtime_seconds):
    """Save reviewer-facing reconstruction outputs and a compact run summary."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / "lumen_mesh.ply"
    lumen_point_cloud_path = output_dir / "lumen_point_cloud.ply"
    branch_point_cloud_path = output_dir / "branch_point_cloud.ply"
    preview_path = output_dir / "segmentation_preview.png"
    summary_path = output_dir / "run_summary.json"

    if not _geometry_has_points(mapper.mesh_near_lumen):
        raise RuntimeError(
            "The mapping run did not produce a lumen mesh, so reviewer "
            "outputs were not saved."
        )
    if not _geometry_has_points(mapper.volumetric_near_point_cloud):
        raise RuntimeError(
            "The mapping run did not produce a lumen point cloud, so reviewer "
            "outputs were not saved."
        )
    if not _geometry_has_points(mapper.volumetric_far_point_cloud):
        raise RuntimeError(
            "The mapping run did not produce a branch point cloud, so reviewer "
            "outputs were not saved."
        )
    if mapper.segmentation_preview is None:
        raise RuntimeError(
            "The mapping run did not produce a segmentation preview, so "
            "reviewer outputs were not saved."
        )

    if not o3d.io.write_triangle_mesh(str(mesh_path), mapper.mesh_near_lumen):
        raise RuntimeError(f"Failed to save lumen mesh: {mesh_path}")
    if not o3d.io.write_point_cloud(
        str(lumen_point_cloud_path), mapper.volumetric_near_point_cloud
    ):
        raise RuntimeError(
            f"Failed to save lumen point cloud: {lumen_point_cloud_path}"
        )
    if not o3d.io.write_point_cloud(
        str(branch_point_cloud_path), mapper.volumetric_far_point_cloud
    ):
        raise RuntimeError(
            f"Failed to save branch point cloud: {branch_point_cloud_path}"
        )
    if not cv2.imwrite(str(preview_path), mapper.segmentation_preview):
        raise RuntimeError(f"Failed to save segmentation preview: {preview_path}")

    model_name = Path(mapper.model_path).name if mapper.model_path else None
    summary = {
        "dataset": Path(mapper.write_folder).name,
        "runtime_seconds": round(float(runtime_seconds), 3),
        "model": model_name,
        "voxel_size": float(mapper.voxel_size),
        "outputs": {
            "lumen_mesh": mesh_path.name,
            "lumen_point_cloud": lumen_point_cloud_path.name,
            "branch_point_cloud": branch_point_cloud_path.name,
            "segmentation_preview": preview_path.name,
        },
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(f"Saved mapping outputs to: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run offline IVUS-EM mapping on a dataset from "
            "Zenodo record 20737792."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-d",
        "--dataset-name",
        choices=VALID_DATASETS,
        help=(
            "Zenodo dataset name, for example patient_1 or sheep_2. "
            "The archive is downloaded and extracted automatically if absent."
        ),
    )
    source.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to an already extracted dataset directory.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=SCRIPT_DIR / "data",
        help=(
            "Parent directory containing extracted named datasets "
            "(default: ./data next to this script)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Mapping configuration YAML file "
            "(default: aortascope_mapping_params.yaml next to this script)."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Optional segmentation-model weights. This overrides model_path in "
            "the YAML configuration. When omitted and the configured model is "
            "not found, the released weights are downloaded automatically to "
            "./deeplumen/weights.weights.h5."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for saved reconstruction outputs. By default, outputs "
            "are written to ./outputs/<dataset> next to this script."
        ),
    )
    download_options = parser.add_mutually_exclusive_group()
    download_options.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Do not download a missing named dataset; require it to already "
            "exist under --data-root."
        ),
    )
    download_options.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "Redownload and re-extract the selected named dataset even if a "
            "local copy already exists."
        ),
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset_path is not None:
        if args.force_download:
            raise ValueError(
                "--force-download can only be used with --dataset-name."
            )

        dataset_path = args.dataset_path

        # dataset_path = args.dataset_path.expanduser().resolve()
        # if not dataset_path.is_dir():
        #     raise FileNotFoundError(
        #         f"Dataset directory does not exist: {dataset_path}"
        #     )
    elif args.no_download:
        dataset_path = (args.data_root / args.dataset_name).expanduser().resolve()
        if not _dataset_layout_exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset is not available under {dataset_path}. "
                "Remove --no-download to retrieve it automatically from Zenodo."
            )
    else:
        dataset_path = download_and_extract_dataset(
            dataset_name=args.dataset_name,
            data_root=args.data_root,
            force_download=args.force_download,
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (SCRIPT_DIR / "outputs" / dataset_path.name).resolve()
    )

    pc_updater = PointCloudUpdater(
        dataset_path=dataset_path,
        config_path=args.config,
        model_path=args.model,
    )

    start_time = time.perf_counter()
    try:
        pc_updater.replay_function()
        runtime_seconds = time.perf_counter() - start_time
        save_mapping_outputs(
            mapper=pc_updater,
            output_dir=output_dir,
            runtime_seconds=runtime_seconds,
        )
    finally:
        pc_updater.vis.destroy_window()


if __name__ == "__main__":
    main()
