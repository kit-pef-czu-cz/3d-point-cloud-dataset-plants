# Annotated 3D Point Cloud Dataset of High-Throughput Plant Scans

Living [repository](https://github.com/kit-pef-czu-cz/3d-point-cloud-dataset-plants) of **3D Point Cloud plant scans**. This dataset provides high-throughput, organ-level annotated 3D point cloud scans of plants, collected using the LeasyScan phenotyping platform.

If you find the the dataset useful, please cite the original paper **Annotated 3D Point Cloud Dataset of Broad-Leaf Legumes Captured by High-Throughput Phenotyping Platform** published in Scientific Data:
```
CITATION
```

## Table of Contents

*   [Dataset Overview](#dataset-overview)
*   [File Structure](#file-structure)
*   [Data acquisition](#data-acquisition)
*   [Raw data preprocessing](#raw-data-preprocessing)
*   [Data Annotation](#data-annotation)
*   [Data Format](#data-format)
*   [License](#license)
*   [Contributing-Collaborating](#contributing-collaborating)
*   [Acknowledgements](#acknowledgements)
*   [Contact](#contact)

## Dataset Overview

This dataset includes annotated 3D point cloud scans of several plant species for various plant organs (e.g., embryonic leaves, petioles, stems, etc.). 
The data was collected using the LeasyScan high-throughput phenotyping platform, which uses **Phenospex PlantEye F600** scanners. The dataset is ideal for use in, e.g., **3D computer vision**, **plant phenotyping** research.

| Name                                           | 	Count |
|------------------------------------------------|--------|
| **Total number of scans**                      | 	223   |
| Scans of common bean specie                    | 	50    |
| Scans of cowpea specie                         | 	45    |
| Scans of lima bean specie                      | 	58    |
| Scans of mungbean specie                       | 	71    |
| **Scans with fully annotated plants**          | 	220   |
| Scans with fully annotated plants using organs | 	141   |
| Scans with unannotated plants using organs     | 	85    |
| Scans containing unannotated plants            | 	3     |
| **Annotated classes**                          | 	5     |
| **Annotated objects (all classes)**            | 	3 712 |
| Annotated objects (Embryonic leaf)             | 	1287  |
| Annotated objects (Leaf)                       | 	1224  |
| Annotated objects (Petiole)                    | 	814   |
| Annotated objects (Stem)                       | 	88    |
| Annotated objects (Plant)                      | 	299   |


## Dataset Structure
````
root/
│
├── data/                                # Contains all point cloud data and annotations
│   ├── Generated cuboid annotations/    # Generated annotations in KITTI (.txt) format for object detection (cuboids)
│   ├── Point clouds/                    # Point cloud data files in .PCD format.
│   ├── Annotation data.csv              # A CSV (and excel) file that contains associations of annotated objects and individual plants in a scan file. A single line in the file represents an individual plant.
│   ├── Raw data.zip                     # Raw data from the scanner. There are always two files (each from a single scanner) for each bar code
│   ├── Segments-ai annotation format.md # description of the segments.ai annotation format 
│   └── Segments-ai annotations.json     # segmentation annotations (point-based) using the abovementioned format from the Segments.ai platform
│
├── code/               # Preprocessing and cuboid generation scripts
│   ├── preprocess.py   # Preprocessing pipeline
│   └── generate\_cuboids.py   # Script for generating cuboids
│
├── LICENSE.md      # Full CC BY-SA 4.0 license
└── README.md       # This documentation in Markdown format
````

## Data acquisition
The presented data were generated using a commercially available PlantEye technology (F600), which is a unique plant phenotyping sensor that combines a 3D scanner with multispectral imaging ([PlantEye F600 multispectral 3D scanner for plants - PHENOSPEX](https://phenospex.com/products/plant-phenotyping/planteye-f600-multispectral-3d-scanner-for-plants/)).
The provided data comes from three regular experimentations in 2022 and 2023 at the ICRISAT field (located in Hyderabad, India). Please see the published paper for details. 

## Raw data preprocessing

The dataset includes a preprocessing code that can be used for the raw point cloud data. The key steps include:

1.  **Rotation** of point clouds to align the plant on the x-plane.
2.  **Merging** merging the point clouds from the two scanners into one file.
3.  **Voxelization** to adjust the resolution of the point cloud.
4.  **Soil Segmentation** to separate plants from soil and trays using AI-based algorithms.

Refer to the published paper for detailed description.

## Data Annotation
The data were annotated using the online platform Segments.ai (https://segments.ai) under an academic license.
Annotations are provided for the following plant organs:

*   Embryonic leaf
*   Leaf
*   Petiole
*   Stem
*   Plant

The Plant class was added for the plants that are, e.g., distorted by wind and do not allow humans to distinguish the plant organs.

## Data Format

* Raw data are proviuded in **.PLY format**; see https://paulbourke.net/dataformats/ply/ for details. 
* Annotated point clouds are provided in **.PCD format**; see https://pcl.readthedocs.io/projects/tutorials/en/latest/pcd_file_format.html for details.
* Annotations:
  * Generated cuboids are using KITTI format; see https://github.com/dtczhl/dtc-KITTI-For-Beginners/blob/master/README.md for details.
  * Segmentation annotations are in the original format from the Segments.ai platform, see `Segments-ai annotation format.md`.

## License

This dataset and associated code are released under the [CC BY-SA 4.0](LICENSE.md).

## Contributing-Collaborating

We welcome any ideas and collaborations! If you want another data for annotation, do not hesitate to contact us. 

## Acknowledgements

This dataset was developed with support from:

* CZU Prague (Czech University of Life Sciences Prague)  
* ICRISAT (International Crops Research Institute for the Semi-Arid Tropics)
*   Phenospex (scanner manufacurer)
*   Segments.ai for annotation, thanks for the free academic license

## Contact

For questions or collaborations, please contact:

* **Jan Masner**: CZU Prague; [masner@pef.czu.cz](mailto:masner@pef.czu.cz) (technical area)
* or **Jana Kholová**: CZU Prague and ICRISAT; [kholova@pef.czu.cz](mailto:kholova@pef.czu.cz) (plant phenotyping)