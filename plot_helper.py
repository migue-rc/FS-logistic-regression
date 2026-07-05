import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
import numpy as np

def plot_sigmoid_interactive(X, y_true,
                             title="Sigmoid Classification",
                             sigmoid_slope=-0.7,
                             sigmoid_center=62):
    # 1. Sigmoid Math
    def sigmoid(x):
        return 1 / (1 + np.exp(sigmoid_slope * (x - sigmoid_center)))

    x_curve = np.linspace(0, 100, 500)
    y_curve = sigmoid(x_curve)

    # 2. Define Slider Ranges (using formatted strings for names)
    thresholds = np.linspace(0, 1, 51)
    x_positions = np.linspace(0, 100, 101)

    fig = go.Figure()

    # 3. Add Base Traces (Static Data)
    # Indices: 0: Rain, 1: No Rain, 2: Curve
    fig.add_trace(go.Scatter(x=X[y_true == 1], y=y_true[y_true == 1],
                             name='Rain (1)', mode='markers', marker=dict(color='blue', opacity=0.4)))
    fig.add_trace(go.Scatter(x=X[y_true == 0], y=y_true[y_true == 0],
                             name='No Rain (0)', mode='markers', marker=dict(color='green', opacity=0.4)))

    mask_initial_green = y_curve < 0.5
    mask_initial_blue = y_curve >= 0.5
    fig.add_trace(go.Scatter(x=x_curve[mask_initial_green], y=y_curve[mask_initial_green],
                             name='Sigmoid (No Rain)', line=dict(color='green', dash='dot')))

    # Index 3: Blue Curve (Above Threshold)
    fig.add_trace(go.Scatter(x=x_curve[mask_initial_blue], y=y_curve[mask_initial_blue],
                             name='Sigmoid (Rain)', line=dict(color='blue', dash='dot')))
    # fig.add_trace(go.Scatter(x=x_curve, y=y_curve, name='Sigmoid', line=dict(color='gray', dash='dot')))


    # 4. Add Moving Traces (Initial State)
    # Index 3: Horizontal Line
    fig.add_trace(go.Scatter(x=[0, 100], y=[0.5, 0.5], mode='lines',
                             line=dict(color='orange', width=3), name='Threshold'))
    # Index 4: Bold Point
    fig.add_trace(go.Scatter(x=[60], y=[sigmoid(60)], mode='markers',
                             marker=dict(size=14,
                                         color='green',
                                         line=dict(width=2, color='black')),
                             name='Current Point'))

    # 5. Build Frames — only the combinations the sliders can actually
    # reach (each slider resets the other axis), and only the traces that
    # change per frame. This is what keeps the exported page small: the
    # full cartesian product would be 51*101 frames of duplicated data.
    def make_frame(t_val, x_val):
        mask_green = y_curve < t_val
        mask_blue = y_curve >= t_val
        y_p = sigmoid(x_val)
        p_color = "blue" if y_p >= t_val else "green"
        return go.Frame(
            data=[
                go.Scatter(x=x_curve[mask_green], y=y_curve[mask_green]),  # Trace 2: Green Curve
                go.Scatter(x=x_curve[mask_blue], y=y_curve[mask_blue]),    # Trace 3: Blue Curve
                go.Scatter(x=[0, 100], y=[t_val, t_val]),                  # Trace 4: Horiz Line
                go.Scatter(x=[x_val], y=[y_p], marker=dict(color=p_color)) # Trace 5: Bold Point
            ],
            traces=[2, 3, 4, 5],
            name=f"t_{t_val}_x_{x_val}"
        )

    frames = [make_frame(round(t, 2), 60) for t in thresholds]
    frames += [make_frame(0.5, int(x_p)) for x_p in x_positions if int(x_p) != 60]

    fig.frames = frames

    # 6. Create Slider Steps
    # Threshold slider resets X to 50
    threshold_steps = []
    for t in thresholds:
        t_label = str(round(t, 2))
        threshold_steps.append({
            "args": [[f"t_{t_label}_x_60"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": t_label,
            "method": "animate"
        })

    # X-Position slider resets Threshold to 0.5
    point_steps = []
    for x_p in x_positions:
        x_label = str(int(x_p))
        point_steps.append({
            "args": [[f"t_0.5_x_{x_label}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": x_label,
            "method": "animate"
        })

    # 7. Layout
    fig.update_layout(
        sliders=[
            {
                "active": int(len(threshold_steps)/2), # Initial threshold 0.5
                "currentvalue": {"prefix": "Threshold Line (Y): "},
                "pad": {"t": 60},
                "steps": threshold_steps,
                "y": 0.05
            },
            {
                "active": int(len(point_steps)/2), # Initial X position 50
                "currentvalue": {"prefix": "Sigmoid Point (X): "},
                "pad": {"t": 160},
                "steps": point_steps,
                "y": -0.15
            }
        ],
        xaxis=dict(range=[0, 100], title="Independent Variable (X)"),
        yaxis=dict(range=[-0.1, 1.1], title="Probability (Y)"),
        title=title,
        height=800,
        margin=dict(b=200), # Space for sliders
        showlegend=False
    )

    return fig

def plot_class(X, y_true, line=None, title=None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=X[y_true == 1],
                   y=y_true[y_true == 1],
                   name='Rain (1)',
                   mode='markers',
                   marker=dict(color='blue'
                               )))
    fig.add_trace(
        go.Scatter(x=X[y_true == 0],
                   y=y_true[y_true == 0],
                   name='No Rain (0)',
                   mode='markers',
                   marker=dict(color='green'
                               )))
    if line:
        steps = np.linspace(0, 100, 50)
        fig.add_trace(go.Scatter(
            x=[line, line],
            y=[0, 1],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='Threshold'
        ))
        fig.add_trace(go.Scatter(x=[line - 2], y=[0.5], text=["Not Rain"],
                                 mode='text', textposition="middle left", name='Left Label',
                                 textfont=dict(color="green", size=14)))

        fig.add_trace(go.Scatter(x=[line + 2], y=[0.5], text=["Rain"],
                                 mode='text', textposition="middle right", name='Right Label',
                                 textfont=dict(color="blue", size=14)))
        frames = []
        for step in steps:
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=X[y_true == 1], y=y_true[y_true == 1]),
                    go.Scatter(x=X[y_true == 0], y=y_true[y_true == 0]),
                    go.Scatter(x=[step, step], y=[-0.1, 1.1]),
                    go.Scatter(x=[step - 2]),
                    go.Scatter(x=[step + 2])
                ],
                name=str(int(round(step, 0)))
            ))
        fig.frames = frames
        sliders = [{
            'active': int(line / 2),
            'currentvalue': {"prefix": "Rain division at ", "suffix": "% of relativity humidity."},
            'pad': {"t": 50},
            'steps': [
                {
                    'args': [
                        [f.name],
                        {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}
                    ],
                    'label': f.name,
                    'method': 'animate'
                } for f in frames
            ]
        }]
        fig.update_layout(
            sliders=sliders,
            xaxis=dict(range=[0, 105]),
            yaxis=dict(range=[-0.1, 1.1]),
            title=title,
            showlegend=False
        )

    return fig


from plotly.subplots import make_subplots

def plot_lr_fit(X, y_true,slope_history, loss_history):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    mu = X.mean()
    sigma = X.std()
    # Dense x-axis for smooth sigmoid curve (normalized + raw for display)
    X_norm_dense = np.linspace(-3, 3, 200)
    X_raw_dense = X_norm_dense * sigma + mu

    # --- Build figure with 2 subplots ---
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=False,
        vertical_spacing=0.22,
        subplot_titles=("Sigmoid fit per epoch", "Loss curve")
    )

    # Scatter: class 0
    fig.add_trace(go.Scatter(
        x=X[y_true == 0], y=y_true[y_true == 0],
        mode="markers",
        name="class 0",
        marker=dict(color="#378ADD", size=10, symbol="circle"),
    ), row=1, col=1)

    # Scatter: class 1
    fig.add_trace(go.Scatter(
        x=X[y_true == 1], y=y_true[y_true == 1],
        mode="markers",
        name="class 1",
        marker=dict(color="#EF9F27", size=10, symbol="circle"),
    ), row=1, col=1)

    # Sigmoid curve (epoch 0) - Using slope_history instead of history
    m0, b0 = slope_history[0]  # Changed: was history[0]["m"], history[0]["b"]
    fig.add_trace(go.Scatter(
        x=X_raw_dense,
        y=sigmoid(m0 * X_norm_dense + b0),
        mode="lines",
        name="sigmoid fit",
        line=dict(color="#534AB7", width=2.5),
    ), row=1, col=1)

    # Loss history line - Using actual loss_history length
    fig.add_trace(go.Scatter(
        x=list(range(len(loss_history))), y=loss_history,  # Changed: dynamic length
        mode="lines",
        name="loss",
        line=dict(color="#1D9E75", width=1.5),
        showlegend=False,
    ), row=2, col=1)

    # Current epoch dot on loss curve
    fig.add_trace(go.Scatter(
        x=[0], y=[loss_history[0]],  # Changed: was loss_vals[0]
        mode="markers",
        name="current epoch",
        marker=dict(color="#1D9E75", size=9),
        showlegend=False,
    ), row=2, col=1)


    # --- Animation frames: subsample epochs (~100 frames). One frame per
    # epoch at 5000 epochs made the exported page tens of MB.
    num_epochs_total = len(slope_history)
    frame_step = max(1, num_epochs_total // 100)
    frame_epochs = list(range(0, num_epochs_total, frame_step))
    if frame_epochs[-1] != num_epochs_total - 1:
        frame_epochs.append(num_epochs_total - 1)
    frames = []
    for i in frame_epochs:
        m_i, b_i = slope_history[i]  # Changed: unpack tuple from slope_history
        loss_i = loss_history[i]     # Changed: get loss from loss_history
        curve_y = sigmoid(m_i * X_norm_dense + b_i)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=X_raw_dense, y=curve_y),   # trace index 2 (sigmoid)
                go.Scatter(x=[i], y=[loss_i]),          # trace index 4 (dot)
            ],
            traces=[2, 4],
            name=str(i),
        ))

    fig.frames = frames


    # --- Slider steps - Use actual number of epochs ---
    num_epochs = len(slope_history)  # Changed: dynamic epoch count
    slider_steps = [
        dict(
            args=[[str(i)], dict(
                frame=dict(duration=0, redraw=False),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=str(i),
            method="animate",
        )
        for i in frame_epochs
    ]

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Epoch: ", font=dict(size=13)),
        pad=dict(t=50, b=10),
        len=0.9,
        x=0.05,
        steps=slider_steps,
    )]


    # --- Play / Pause buttons ---
    updatemenus = [dict(
        type="buttons",
        showactive=False,
        direction="left",
        x=0.05, xanchor="right",
        y=-0.08, yanchor="top",
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=40, redraw=False),
                    fromcurrent=True,
                    transition=dict(duration=0),
                )],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                    transition=dict(duration=0),
                )],
            ),
        ],
    )]


    # --- Layout ---
    fig.update_layout(
        height=650,
        updatemenus=updatemenus,
        sliders=sliders,
        legend=dict(orientation="h", y=1.14, x=0, yanchor="bottom"),
        margin=dict(t=100, b=80),
    )

    fig.update_xaxes(title_text="humidity (%)", range=[28, 102], row=1, col=1)
    fig.update_yaxes(title_text="probability", range=[-0.1, 1.1], row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=2, col=1)
    fig.update_yaxes(title_text="loss", row=2, col=1)

    return fig
