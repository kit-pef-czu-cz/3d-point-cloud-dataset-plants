Configurations are based on the default configurations for custom datasets (in case of SECONDNet) and KITTI (in case of PointRCNN). Hyperparameters were only configured to accomodate our dataset, no other changes were made.

`base_second_dat.yaml`, based on OpenPCDet's `dataset_configs/custom_dataset.yaml`:
* `POINT_CLOUD_RANGE`: Accomodating the range of our scans
* `MAP_CLASS_TO_KITTI`: Necessary
* `DATA_SPLIT` and `INFO_PATH`: Removed, replaced with loading train/val/test splits from code directly
* `POINT_FEATURE_ENCODING`: Removed `intensity`
* `DATA_AUGMENTOR.AUG_CONFIG_LIST`: Emptied - no augmentations used
* `DATA_PROCESSOR.transform_points_to_voxels`:
	* `VOXEL_SIZE`: Resolution set to [2, 2, 2] - lower numbers (higher resolutions) result in individual disjointed voxels
	* `MAX_POINTS_PER_VOXEL`: Increased to 50
	* `MAX_NUMBER_OF_VOXELS`: Set to 24000 and 80000

`base_second_mdl.yaml`, based on OpenPCDet's `custom_models/second.yaml`:
* `CLASS_NAMES`: Necessary
* `MODEL.MAP_TO_BEV.NUM_BEV_FEATURES`: Necessary based on cloud range/voxel size settings
* `DENSE_HEAD.ANCHOR_GENERATOR_CONFIG`: Only one Plant class.
	* `anchor_sizes`: Original sizes from the SECOND paper use mean sizes of respective classes. `[[57.4, 54.8, 36.5]]` used here are the mean dimensions of the dataset's Plant class bounding boxes
	* 'anchor_rotations': Replaced with [0], rotations are ignored.
	
`base_pointrcnn_mdl.yaml`, based on OpenPCDet's `kitti_models/pointrcnn.yaml`:
* `BACKBONE_3D.SA_CONFIG.RADIUS`, `POINT_HEAD.TARGET_CONFIG.GT_EXTRA_WIDTH`, `POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.mean_size`,`ROI_HEAD.SA_CONFIG.RADIUS`,
	* All values multiplied by 24, as that's the mean factor of mean bounding box dimensions between KITTI and our dataset