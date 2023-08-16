from itertools import combinations
import torch


def filter_overlapping(bboxes_conf_cls, threshold):
    if len(bboxes_conf_cls) == 1:
        return bboxes_conf_cls

    merged_flag = False
    # Get all pairs of combinations of inp_bboxes
    for p1, p2 in list(combinations(zip(bboxes_conf_cls, range(len(bboxes_conf_cls))), 2)):
        bb1, bb2 = p1[0], p2[0]
        i1, i2 = p1[1], p2[1]
        iosa = bbox_iosa(bb1[0], bb2[0]).item()
        
        if iosa > threshold:
            # get the bounding box with largest confidence.
            # or if equal confidence, prioritise the one with 'confidence_class': 'high'
            if bb1[1] > bb2[1]:
                bbox = bb1
            elif bb1[1] == bb2[1]:
                bbox = bb2
                if bb1[2] == 'high':
                    bbox = bb1
            else:
                bbox = bb2
                
            # discard the smaller box
            bboxes_conf_cls[i1] = bbox
            del bboxes_conf_cls[i2]
            merged_flag = True
            break
        
    if not len(bboxes_conf_cls) == 1 and merged_flag:
        filter_overlapping(bboxes_conf_cls, threshold)
        
    return bboxes_conf_cls


def _format_results(res):
    return [{
        'bbox':  x[0][0].tolist() + [x[1]],
        'confidence_class': x[2]
        } for x in res]
   

def overlap_postprocessing(detections, mode, iou_thres):
    """
    Process overlapping detections from multiple passes.
    """
    res = {}
    for cls_name, dets in detections.items():
        if mode == 'max': # Take the bounding box with highest confidence.
            res[cls_name] = _format_results(
                                filter_overlapping([
                                    (torch.FloatTensor([x['bbox'][:4]]), # bbox
                                     x['bbox'][4], # confidence
                                     x['confidence_class']) 
                                    for x in dets], iou_thres))
        elif mode == 'merge': # Merge overlapping bounding boxes.
            inp_bboxes = [torch.FloatTensor([x['bbox'][:4]]) for x in dets]
            inp_confs = [x['bbox'][4] for x in dets]
            inp_conf_class = [x['confidence_class'] for x in dets]
            res[cls_name] = _format_results(
                                zip(*merge_postprocessing(
                                    inp_bboxes, 
                                    inp_confs, 
                                    threshold=iou_thres, 
                                    inp_conf_class=inp_conf_class)))
        else:
            raise ValueError('Invalid mode for multi pass overlap '+
                             'postprocessing, choose from "max" or "merge"')
    return res



def merge_postprocessing(inp_bboxes, inp_confidences, threshold, inp_conf_class=None):
    if len(inp_bboxes) == 1:
        return inp_bboxes, inp_confidences, inp_conf_class

    merged_flag = False
    if inp_conf_class is None:
        inp_conf_class = [None] * len(inp_bboxes)

    # Get all pairs of combinations of inp_bboxes
    for p1, p2 in list(combinations(zip(inp_bboxes, inp_confidences, inp_conf_class, range(len(inp_bboxes))), 2)):
        bb1, bb2 = p1[0], p2[0]
        c1, c2 = p1[1], p2[1]
        l1, l2 = p1[2], p2[2] # Label
        i1, i2 = p1[3], p2[3]
        bb_pair_iosa = bbox_iosa(bb1, bb2).item()

        if bb_pair_iosa > threshold:
            # get the largest bounding box
            bb1_gt_bb2 = compare_bounding_box_areas(bb1[0], bb2[0])
            bbox = bb1 if bb1_gt_bb2 else bb2
            conf = c1 if bb1_gt_bb2 else c2
            lbl  = l1 if bb1_gt_bb2 else l2
            
            # discard the smaller box
            inp_bboxes[i1] = bbox
            inp_confidences[i1] = conf
            inp_conf_class[i1] = lbl

            del inp_bboxes[i2]
            del inp_confidences[i2]
            del inp_conf_class[i2]
            merged_flag = True
            break

    if not len(inp_bboxes) == 1 and merged_flag:
        merge_postprocessing(inp_bboxes, inp_confidences, threshold, inp_conf_class=inp_conf_class)

    return inp_bboxes, inp_confidences, inp_conf_class


def bbox_iosa(box1, box2, eps=1e-7):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + eps
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Get area of smaller box
    smaller_box_area = box2_area if box1_area > box2_area else box1_area

    return inter / smaller_box_area


def compare_bounding_box_areas(bb_1, bb_2):
    area1 = (bb_1[2] - bb_1[0]) * (bb_1[3] - bb_1[1])
    area2 = (bb_2[2] - bb_2[0]) * (bb_2[3] - bb_2[1])
    return area1 > area2
