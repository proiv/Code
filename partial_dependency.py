def partial_dependency(model, X, y, feature_ids = [], f_id = -1):

    """
    Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    1. Sample a grid of values of a predictor.
    2. For each value, replace every row of that predictor with this value, calculate the average prediction.
    3. Return a grid and average predictors
    """

    X_temp = X.copy()

    grid = np.linspace(np.percentile(X_temp.iloc[:, f_id], 0.1),
                       np.percentile(X_temp.iloc[:, f_id], 99.5),
                       50)
    y_pred = np.zeros(len(grid))

    if len(feature_ids) == 0 or f_id == -1:
        print ('Input error!')
        return
    else:
        for i, val in enumerate(grid):

            X_temp.iloc[:, f_id] = val
            data = xgb.DMatrix(X_temp, feature_names = df_columns)

            y_pred[i] = np.average(model.predict(data))

    return grid, y_pred
