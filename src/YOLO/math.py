"""
Script responsible for holding utility functions
"""

Epsilon = 1E-6


def iou(box1,box2):
    """
    Computes Intersection over Union for Yolo
    (x, y) Center of a box
    (w, h) Width and height oof a box
    c Confidence score
    :param box1: Tensor of shape (x, y, w, h, c)
    :param box2: Tensor of shape (x, y, w, h, c)
    :return: IoU score
    """
    # Compute the corners
    b1x1 = box1[1] - box1[3] / 2
    b1y1 = box1[1] + box1[4] / 2
    b1x2 = box1[1] + box1[3] / 2
    b1y2 = box1[1] - box1[4] / 2
    b2x1 = box2[1] - box2[3] / 2
    b2y1 = box2[1] + box2[4] / 2
    b2x2 = box2[1] + box2[3] / 2
    b2y2 = box2[1] - box2[4] / 2

    # Compute coordinates of the intersection
    x1 = max(b1x1,b2x1)
    y1 = max(b1y1,b2y1)
    x2 = min(b1x2,b2x2)
    y2 = min(b1y2,b2y2)

    # Compute interesting values
    intersection = max(x2 - x1,0) * max(y2 - y1,0)
    union = box1[3] * box1[4] + box2[3] * box2[4] - intersection
    return intersection / (union + Epsilon)


def nms(boxes,threshold_iou,threshold):
    """
    Compute Non-Maximum Separation
    :param boxes: List of predicted boxes in the format (x, y, w, h, c)
    :param threshold_iou: The threshold to remove overlapping boxes
    :param threshold: The threshold to keep interesting boxes
    :return: List of boxes to keep after nms
    """
    # Sort boxes according to their confidence
    sorted_boxes = [box for box in boxes if box[4] > threshold]
    sorted_boxes = sorted(sorted_boxes,key=lambda x: x[4],reverse=True)
    keep = []
    while sorted_boxes:
        candidate = sorted_boxes.pop(0)

        boxes = [
            box
            for box in boxes
            if candidate[4] != box[4] or iou(candidate,box) < threshold_iou
        ]
        keep.append(candidate)
    return keep


def mAP(prediction,ground_truth,iou_threshold,n_classes=20):
    """
    Compute Mean Average Precision
    :param prediction: Predicted boxes
    :param ground_truth: Ground truth boxes
    :param iou_threshold: IoU threshold
    :param n_classes: Number of classes
    :return: The mean average precision score across all classes given a specific IoU threshold
    """
    # List of the precisions
    average_precisions = []
