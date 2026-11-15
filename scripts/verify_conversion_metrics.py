"""Analytic checks for conversion metrics, including missed predictions and center/IoU distinction."""
import numpy as np
from evaluate_conversion_matrix import score_predictions, compare_outputs


def main():
    record={'label':'0 0.5 0.5 0.1 0.1\n','image':'analytic.png'}
    perfect=np.array([[576.,460.8,704.,563.2,.9,0.]])
    result=score_predictions([record],[perfect])
    assert result['thresholds']['0.25']['center5_f1']==1
    assert result['thresholds']['0.25']['matched_error_px_p50_p95_p99']==[0,0,0]
    assert result['box_mAP50']>.99
    shifted=perfect.copy()
    shifted[:,[0,2]]+=20
    result=score_predictions([record],[shifted])
    assert result['thresholds']['0.25']['center5_recall']==0
    assert result['box_mAP50']>.99  # overlap can be high while center is wrong
    empty=np.empty((0,6))
    result=score_predictions([record],[empty])
    assert result['thresholds']['0.25']['center5_recall']==0
    assert result['thresholds']['0.25']['matched_error_px_p50_p95_p99'] is None
    comparison=compare_outputs([record],[perfect],[empty])
    assert comparison['new_gt_misses_at_5px']==1
    assert comparison['unmatched_reference_predictions']==1
    slight=perfect.copy()
    slight[:,[0,2]]+=.1
    comparison=compare_outputs([record],[perfect],[slight])
    assert comparison['new_gt_misses_at_5px']==0
    assert abs(comparison['prediction_center_shift_px_p50_p95_p99_max'][-1]-.1)<1e-8
    print('Analytic center, IoU, empty-output and backend-drift checks passed.')


if __name__=='__main__':
    main()
