import plotly.express as px

def plot_predicted_vs_labels(df):

    max_value = max(max(df["y_pred"]), max(df["y_true"]))

    plot = px.scatter(
        df,
        x = "y_pred",
        y = "y_true",
        template="plotly_white",
        labels={
            "y_pred": "Predicted value",
            "y_true": "True value"
                 },
        title="Cyanobacteria monitoring prediction with S2A")

    return plot.update_layout(shapes=[
        {"type": "line",
        "y0":0,
        "y1":max_value,
        "x0":0,
        "x1": max_value}])