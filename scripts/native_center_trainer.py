"""Native 1280x1024 batches, FP32 center-based validation, no site-package edits."""
from copy import copy
import numpy as np
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops

from diagnose_ball_centers import match_pairs


class NativeCenterValidator(DetectionValidator):
    def init_metrics(self,model):
        super().init_metrics(model)
        self.center_gt=self.center_pred=self.center_tp5=self.center_tp10=0
        self.center_errors=[]
        self.actual_shapes=set()

    def preprocess(self,batch):
        shape=tuple(batch['img'].shape[-2:])
        if shape!=(1024,1280):
            raise RuntimeError(f'Native validation shape changed: {shape}')
        self.actual_shapes.add(shape)
        return super().preprocess(batch)

    def update_metrics(self,preds,batch):
        for si,pred in enumerate(preds):
            pbatch=self._prepare_batch(si,batch)
            gt=ops.scale_boxes(pbatch['imgsz'],pbatch['bboxes'].clone(),pbatch['ori_shape'],pbatch['ratio_pad']).cpu().numpy()
            boxes=pred['bboxes'][pred['conf']>=.25].clone()
            boxes=ops.scale_boxes(pbatch['imgsz'],boxes,pbatch['ori_shape'],pbatch['ratio_pad']).cpu().numpy()
            d=np.linalg.norm((gt[:,None,:2]+gt[:,None,2:])/2-(boxes[None,:,:2]+boxes[None,:,2:])/2,axis=-1)
            self.center_gt+=len(gt);self.center_pred+=len(boxes)
            self.center_tp5+=len(match_pairs(d,5))
            pairs=match_pairs(d,10)
            self.center_tp10+=len(pairs)
            self.center_errors.extend(distance for _,_,distance in pairs)
        super().update_metrics(preds,batch)

    def get_stats(self):
        stats=super().get_stats()
        p=self.center_tp5/max(self.center_pred,1)
        r=self.center_tp5/max(self.center_gt,1)
        errors=np.percentile(self.center_errors,[50,95,99]).tolist() if self.center_errors else [10.0]*3
        self.center_result={'gt':self.center_gt,'predictions':self.center_pred,'center5_matches':self.center_tp5,
            'center10_matches':self.center_tp10,'center5_precision':p,'center5_recall':r,
            'center5_f1':2*p*r/max(p+r,1e-12),'matched_error_px_p50_p95_p99':errors if self.center_errors else None,
            'actual_shapes_height_width':[list(shape) for shape in sorted(self.actual_shapes)],
            'conf':.25,'precision':'FP32','missing_gt_center5':self.center_gt-self.center_tp5}
        stats.update({'metrics/center5_precision':p,'metrics/center5_recall':r,
            'metrics/center5_F1':self.center_result['center5_f1'],
            'metrics/center_error_p50':errors[0],'metrics/center_error_p95':errors[1],
            'metrics/center_error_p99':errors[2],'fitness':self.center_result['center5_f1']})
        return stats


class NativeCenterTrainer(DetectionTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        dataset=super().build_dataset(img_path,mode,batch)
        # The installed default validator adds a half-stride padding margin.
        # Uniform 1280x1024 originals need no padding at stride32.
        if not dataset.rect or dataset.imgsz!=1280:
            raise RuntimeError('Native suite requires rect=True and imgsz=1280')
        dataset.pad=0.0
        dataset.batch_shapes[:]=np.array([1024,1280],dtype=int)
        return dataset

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode=='val':
            batch_size=min(batch_size,self.args.batch,8)
        return super().get_dataloader(dataset_path,batch_size,rank,mode)

    def preprocess_batch(self,batch):
        if tuple(batch['img'].shape[-2:])!=(1024,1280):
            raise RuntimeError(f'Native training shape changed: {tuple(batch["img"].shape)}')
        self.observed_train_shape=list(batch['img'].shape)
        return super().preprocess_batch(batch)

    def get_validator(self):
        return NativeCenterValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)

    def validate(self):
        previous_amp=self.amp
        try:
            # Training uses AMP; selection is explicitly full-precision FP32.
            self.amp=False
            metrics=self.validator(self)
        finally:
            self.amp=previous_amp
        fitness=float(self.validator.center_result['center5_f1'])
        metrics.pop('fitness',None)
        if self.best_fitness is None or fitness>self.best_fitness:
            self.best_fitness=fitness
        return metrics,fitness
