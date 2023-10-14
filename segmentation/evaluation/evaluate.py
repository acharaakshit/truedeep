import os
import cv2
import numpy as np
import argparse

def calculate_metrics(predictions_folder, gt_folder, threshold):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for filename in os.listdir(predictions_folder):
        if filename.endswith('.png'):
            prediction_path = os.path.join(predictions_folder, filename)
            mask_path = os.path.join(gt_folder, filename)

            prediction = cv2.imread(prediction_path, 0) / 255
            mask = cv2.imread(mask_path, 0) / 255
            
            # threshold
            prediction = (prediction > threshold).astype(np.uint8)

            tp = np.sum(np.logical_and(prediction > 0, mask > 0))
            fp = np.sum(np.logical_and(prediction > 0, mask == 0))
            fn = np.sum(np.logical_and(prediction == 0, mask > 0))

            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

def find_optimal_threshold(predictions_folder, gt_folder, threshold_range=(0, 1, 0.01)):
    best_threshold = 0
    best_f1_score = 0

    for threshold in np.arange(threshold_range[0], threshold_range[1], threshold_range[2]):
        precision, recall, f1_score = calculate_metrics(predictions_folder, gt_folder, threshold)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_f1_score, best_precision, best_recall

def main():
    parser = argparse.ArgumentParser(description="Compute precision, recall, and F1 score for image segmentation predictions.")
    parser.add_argument("--prediction_path", required=True, help="Path to the predictions folder.")
    parser.add_argument("--gt_path", required=True, help="Path to the masks folder.")
    args = parser.parse_args()

    predictions_folder = args.prediction_path
    gt_folder = args.gt_path

    best_threshold, ods_f1_score, ods_precision, ods_recall = find_optimal_threshold(predictions_folder, gt_folder)
    
    print("Optimal Dataset Scale (ODS) F-score: {:.4f}".format(ods_f1_score))
    print("ODS Precision: {:.4f}".format(ods_precision))
    print("ODS Recall: {:.4f}".format(ods_recall))
    print("ODS Threshold: {:.2f}".format(best_threshold))

    precision, recall, f1_score = calculate_metrics(predictions_folder, gt_folder, 0.5)

    print("Standard Precision: {:.4f}".format(precision))
    print("Standard Recall: {:.4f}".format(recall))
    print("Standard F1 Score: {:.4f}".format(f1_score))

if __name__ == '__main__':
    main()
