"""Analytic checks of native validator center selection, threshold and missed counts."""
from unittest.mock import patch
from types import SimpleNamespace


def main():
    import torch
    from native_center_trainer import NativeCenterValidator
    from ultralytics.models.yolo.detect.val import DetectionValidator
    v=NativeCenterValidator.__new__(NativeCenterValidator)
    v.center_gt=v.center_pred=v.center_tp5=v.center_tp10=0
    v.center_errors=[]
    v.actual_shapes={(1024,1280)}
    gt=torch.tensor([[90.,90.,110.,110.]])
    v._prepare_batch=lambda si,batch:{'imgsz':(1024,1280),'ori_shape':(1024,1280),
        'ratio_pad':((1.,1.),(0.,0.)),'bboxes':gt}
    def prediction(box,conf):
        return {'bboxes':torch.tensor([box]),'conf':torch.tensor([conf]),'cls':torch.tensor([0.])}
    with patch.object(DetectionValidator,'update_metrics',lambda *args:None),patch.object(DetectionValidator,'get_stats',lambda self:{'fitness':0.}):
        v.update_metrics([prediction([97.,97.,103.,103.],.9)],{})
        assert v.center_tp5==1 and v.center_errors==[0.]
        v.update_metrics([prediction([97.,97.,103.,103.],.249)],{})
        assert v.center_gt==2 and v.center_pred==1 and v.center_tp5==1
        v.update_metrics([prediction([103.,97.,109.,103.],.9)],{})
        assert v.center_tp5==1 and v.center_tp10==2
        stats=v.get_stats()
        assert abs(stats['fitness']-.4)<1e-12
        assert v.center_result['missing_gt_center5']==2
        assert v.center_result['gt']==3 and v.center_result['predictions']==2
    print('Passed: tight/loose boxes with same center; confidence cutoff; 5px/10px separation; F1-based fitness; miss counts.')


if __name__=='__main__':
    main()
