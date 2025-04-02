import numpy as np

def calculate_accuracy(y_true, y_pred):
    """
    Tính Accuracy (độ chính xác tổng thể) sử dụng NumPy.
    y_true: mảng hoặc danh sách chứa nhãn thật.
    y_pred: mảng hoặc danh sách chứa nhãn dự đoán.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def calculate_recall(y_true, y_pred, positive=1):
    """
    Tính Recall cho lớp dương tính (mặc định là 1) sử dụng NumPy.
    Recall = TP / (TP + FN)
    
    y_true: mảng hoặc danh sách chứa nhãn thật.
    y_pred: mảng hoặc danh sách chứa nhãn dự đoán.
    positive: giá trị của lớp dương tính (mặc định là 1).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Tính số True Positives (TP)
    TP = np.sum((y_true == positive) & (y_pred == positive))
    # Tính số False Negatives (FN)
    FN = np.sum((y_true == positive) & (y_pred != positive))
    
    return TP / (TP + FN) if (TP + FN) > 0 else 0

if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 1, 1, 0, 1, 1, 0]
    
    acc = calculate_accuracy(y_true, y_pred)
    rec = calculate_recall(y_true, y_pred, positive=1)
    
    print("Accuracy:", acc)
    print("Recall:", rec)
