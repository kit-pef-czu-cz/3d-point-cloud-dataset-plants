# 3D point cloud
Derived from [Segments.ai documentation](https://docs.segments.ai/reference/label-types#id-3d-point-cloud)
## Segmentation label
The annotations array contains the different objects ("annotations") in the label with their category (the `category_id` should correspond to an id defined in the categories).

The `point_annotations` array contains the object/annotation id for each point in the point cloud. The order of the ids in this array is the same as the order of the points in the point cloud.

```
{
  "format_version": "0.1",
  "annotations": [
    {
      "id": 1, // the object id
      "category_id": 1 // the category id
    },
    {
      "id": 2, 
      "category_id": 1
    },
    {
      "id": 3, 
      "category_id": 4
    }
  ],
  "point_annotations": [0, 0, 0, 3, 2, 2, 2, 1, 3...], // refers to object ids
}
```