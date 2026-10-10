from sklearn.model_selection import train_test_split


def split_data(
    X,
    y,
    test_size=0.20,
    validation_size=0.20,
    random_state=42,
):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
        random_state=random_state,
        stratify=y,
    )

    validation_fraction = (
        validation_size / (test_size + validation_size)
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_fraction,
        random_state=random_state,
        stratify=y_temp,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )
