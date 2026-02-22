import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from data_loader import load_data
from preprocessing import preprocess
from model import build_model


def main():
    config = yaml.safe_load(open("config/config.yaml"))

    df = load_data("data/sales.csv")
    df = preprocess(df)

    X = df[["ad_budget", "customers"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)

    print("MAE:", error)


if __name__ == "__main__":
    main()
