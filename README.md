# Annotated 3D Point Cloud Dataset of High-Throughput Plant Scans

Living [repository](https://github.com/kit-pef-czu-cz/3d-point-cloud-dataset-plants) of **3D Point Cloud plant scans**. This dataset provides high-throughput, organ-level annotated 3D point cloud scans of plants, collected using the LeasyScan phenotyping platform.

The original, fixed repository can be found at Figshare: https://doi.org/10.6084/m9.figshare.28270742 

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
*   [Baseline evaluation on object detection models](#baseline-evaluation-on-object-detection-models)
*   [License](#license)
*   [Contributing-Collaborating](#contributing-collaborating)
*   [Acknowledgements](#acknowledgements)
*   [Contact](#contact)

## Dataset Overview

This dataset includes annotated 3D point cloud scans of several plant species for various plant organs (e.g., embryonic leaves, petioles, stems, etc.). 
The data was collected using the LeasyScan high-throughput phenotyping platform, which uses **Phenospex PlantEye F600** scanners. The dataset is ideal for use in, e.g., **3D computer vision**, **plant phenotyping** research.

| Name                                               | 	Count |
|----------------------------------------------------|--------|
| **Total number of scans**                          | 	223   |
| Scans of common bean specie                        | 	50    |
| Scans of cowpea specie                             | 	45    |
| Scans of lima bean specie                          | 	58    |
| Scans of mungbean specie                           | 	71    |
| **Scans with all plants annotated using organs**   | 	141   |
| Scans containing plants unannotated using organs   | 	85    |
| Scans containing some unannotated plants           | 	3     |
| **Annotated classes**                              | 	5     |
| **Annotated objects (all classes)**                | 	3 712 |
| Annotated objects (Embryonic leaf)                 | 	1287  |
| Annotated objects (Leaf)                           | 	1224  |
| Annotated objects (Petiole)                        | 	814   |
| Annotated objects (Stem)                           | 	88    |
| Annotated objects (Plant)                          | 	299   |


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
│   ├── Segments-ai annotations.json     # segmentation annotations (point-based) using the abovementioned format from the Segments.ai platform
│   └── MIAPPE_data.xlsx                 # MIAPPE-compliant (Minimum Information About a Plant Phenotyping Experiment) data sheet including mapping to the Annotation data.csv file.
│
├── code/                      # Preprocessing and cuboid generation scripts
│   ├── Preprocess/            # Preprocessing pipeline
│   └── Cuboids generation/    # Script for generating cuboids in KITTI format
│
├── Baseline evaluation/      # Baseline evaluation on SECOND and PointRCNN models - code and detailed results using nested cross-validation
|
├── LICENSE.md      # Full CC BY 4.0 license
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

Annotations for plant organs to track their assignment to individual plants are in the `Annotation data.csv` file. It contains IDs of annotated objects. A single line represents an individual plant and its organs. The following table provides a description column.

| Column name        | Content description                                                                                                                                                                |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Specie             | Name of the plant specie that the file contains.                                                                                                                                    |
| Exp. num.          | Number of experiment, under which the scan was obtained at ICRISAT.                                                                                                                  |
| Bar code           | Identification of a section within the experiment (position in the LeasyScan platform).                                                                                              |
| Tray               | Identification of the tray within the section.                                                                                                                                      |
| Date time          | Timestamp of the scan in format YYYYMMDDTHHMMSS. The “T” is a divider.                                                                                                                |
| Full-Part-Organs   | “Full” determines that all organs of all plants in the scan were fully annotated. “Part” means the scan contains plant(s) where it is not possible to recognize their organs.         |
| Full-Part-Plants   | “Full” determines whether all plants in the scan were annotated at least using the Plant class. “Part” means there are two or more plants in the scan that overlap and can’t be distinguished from each other. |
| File name          | Name of the file in the provided dataset. The name consists of the following columns, divided by dash (“-”): Exp. Num., Bar code, Tray, Date time.                                     |
| Obj ID X           | Multiple columns named “Obj. ID X” contain IDs of objects (annotated classes) that belong to one plant.                                                                              |


## Data Format

* Raw data are provided in **.PLY format**; see https://paulbourke.net/dataformats/ply/ for details. 
* Annotated point clouds are provided in **.PCD format**; see https://pcl.readthedocs.io/projects/tutorials/en/latest/pcd_file_format.html for details.
* Annotations:
  * Generated cuboids are using KITTI format; see https://github.com/dtczhl/dtc-KITTI-For-Beginners/blob/master/README.md for details.
  * Segmentation annotations are in the original format from the Segments.ai platform, see `Segments-ai annotation format.md`.

## Baseline evaluation on object detection models
We conducted baseline experiments to assess the utility and applicability of the presented dataset using two standard object detection architectures: SECOND, which operates on voxel grids, and PointRCNN, which processes raw points. The codebase utilized the OpenPCDet library (https://github.com/open-mmlab/OpenPCDet) with minor modifications tailored to our dataset. For details on the training procedure see the paper. For reproducing the results, see `Baseline evaluation/README.md`.  

## License

This dataset is released under the [CC BY 4.0](LICENSE.md). The associated source code (`Code` folder) is released under Apache 2.0 license (LICENSE-code.md). 

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
* or **Jana Kholová**: CZU Prague and ICRISAT (formerly); [kholova@pef.czu.cz](mailto:kholova@pef.czu.cz) (plant phenotyping)