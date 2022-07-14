from catboost import CatBoostClassifier


def train_model(train_data, target_grid, new_train: bool = True, model=None):
    if new_train:
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
        )
    x_train = train_data.reshape(-1, train_data.shape[1])
    y_train = target_grid.reshape(-1)
    model.fit(x_train, y_train)
    return model
