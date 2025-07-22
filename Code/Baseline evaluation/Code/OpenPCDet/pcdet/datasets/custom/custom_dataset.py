import copy
import pickle
import os

import numpy as np

# from ...ops.roiaware_pool3d import roiaware_pool3d_utils
# from ...utils import box_utils, common_utils
# from ..dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils
from pcdet.datasets.dataset import DatasetTemplate


class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, split='train'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.split = split
        self.fill_sample_id_list()
        self.custom_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI
        
    def fill_sample_id_list(self):
        self.sample_id_list = []
        template = os.path.join(self.root_path, 'ImageSets', "%.txt")
        for split_path in self.get_split_aug_names(template):
            if not os.path.exists(split_path):
                continue
            self.sample_id_list.extend([x.strip() for x in open(split_path).readlines()])
        if len(self.sample_id_list) == 0:
            self.sample_id_list = None
        else:
            self.sample_id_list.sort()
    
    def get_split_aug_names(self, path_template: str):
        def make_path(name):
            return path_template.replace("%", name)
            # return os.path.join(self.root_path, 'ImageSets', (name + '.txt'))
        result = [make_path(self.split)]
        if self.split == "test":
            return result
        if self.split == "val" and not self.dataset_cfg.CULS_AUGMENTS_VAL:
            return result
        for aug in self.dataset_cfg.CULS_AUGMENTS:
            result.append(make_path(f"{self.split}_{aug}"))
        return result

    def include_data(self, mode):
        # self.logger.info(f'Loading custom dataset, {self.split} split.')
        template = os.path.join(self.root_path, "custom_infos_%.pkl")
        info_paths = self.get_split_aug_names(template)
        custom_infos = []
        for info_path in info_paths:
            if not os.path.exists(info_path):
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)
        custom_infos = sorted(custom_infos, key=lambda d: d['point_cloud']['lidar_idx'])
        self.custom_infos.extend(custom_infos)
        self.logger.info(
            "Loaded split " +
            self.split.ljust(5, ' ') +
            " with " +
            str(len(custom_infos)).rjust(4, ' ') +
            " samples and " +
            (str(len(info_paths)-1).rjust(2, ' ') if len(info_paths) > 1 else "no") +
            " augmentations."
        )
        # self.logger.info(f'Total samples: {len(custom_infos)} ({len(info_paths)-1} augmentations)')

    def get_label(self, idx):
        label_file = self.root_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
        assert lidar_file.exists()
        point_features = np.load(lidar_file)
        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
    
    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        infos = []
        for sample_idx in sample_id_list:
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations
            infos.append(info)
        return infos

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str,
                        help="Path to the dataset config YAML file.")
    parser.add_argument("-d", type=str, default="custom",
                        help="Name of the dataset folder under 'data/'. Default: '%(default)s'")
    parser.add_argument("-c", type=str, nargs="+", default="Plant",
                        help="Dataset class names. Default: '%(default)s'")
    args = parser.parse_args()
    
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    print(ROOT_DIR)
    create_custom_infos(
        dataset_cfg=dataset_cfg,
        class_names=args.c if type(args.c) is list else [args.c],
        data_path=ROOT_DIR / 'data' / args.d,
        save_path=ROOT_DIR / 'data' / args.d,
    )

def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=1):
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
    splits = os.listdir(os.path.join(save_path, "ImageSets"))
    for split_file in splits:
        split_name = split_file[:-4]
        split_filename = save_path / ('custom_infos_%s.pkl' % split_name)
        dataset.set_split(split_name)
        custom_infos = dataset.get_infos(
            class_names, num_workers=workers, has_label=True, num_features=num_features
        )
        with open(split_filename, 'wb+') as f:
            pickle.dump(custom_infos, f)
        if split_name.startswith("train"):
            dataset.create_groundtruth_database(split_filename, split=split_name)


if __name__ == "__main__":
    import argparse, yaml
    from pathlib import Path
    from easydict import EasyDict
    main()
    