from misc import load_data, preprocess_data, train_model, evaluate_model
from sklearn.kernel_ridge import KernelRidge

if __name__ == "__main__":
    # 1. Load Data
    df = load_data()
    
    # 2. Preprocess Data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 3. Initialize & Train Model
    model = KernelRidge()
    trained_model = train_model(model, X_train, y_train)
    
    # 4. Evaluate Model
    mse = evaluate_model(trained_model, X_test, y_test)
    print(f"Kernel Ridge Average MSE: {mse:.4f}")