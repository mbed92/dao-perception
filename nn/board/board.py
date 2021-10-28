import tensorflow as tf


def reset_metrics(metrics: list):
    for m in metrics:
        m.reset_states()
    return metrics


def check_best_metric(metric, current_best, step, metrics_to_display: list = None):
    save_model = False
    if metric.result().numpy() < current_best:
        save_model = True

        current_best = metric.result().numpy()
        print("\n\nStep {}. Current best metric: {}".format(step, current_best))

        if metrics_to_display is not None:
            [print("{}: {}".format(m.name, m.result().numpy())) for m in metrics_to_display]

    return save_model, current_best


def add_to_tensorboard(metrics: dict, step: int, prefix: str):
    for key in metrics:
        if metrics[key] is None:
            return

        if key in ["scalars"]:
            for m in metrics[key]:
                tf.summary.scalar('{}/{}'.format(prefix, m.name), m.result().numpy(), step=step)

        if key in ["images"]:
            for i, img in enumerate(metrics[key]):
                if img is not None:
                    tf.summary.image('{}/{}/{}'.format(prefix, key, i), img, step=step, max_outputs=1)
