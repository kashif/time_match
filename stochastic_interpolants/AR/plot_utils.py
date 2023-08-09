import torch
import seaborn as sns
import matplotlib.pyplot as plt

def plot(original, predictions, data_fn, ode=False, name=None):
    bsz, T = original.shape
    n_samples = predictions.shape[0]
    vis_samples = int(bsz ** 0.5)

    y_avg = data_fn(original, 1).view(bsz, T)

    x_single = torch.arange(T)
    x = x_single
    if not ode:
        x = x_single.view(1, -1).repeat(n_samples, 1).reshape(-1)
        
    for i in range(bsz):
        fig, ax = plt.subplots(figsize=(10, 10))        
        sns.lineplot(x=x, y=predictions[:, i, :].reshape(-1), errorbar=('ci', 95), ax=ax, c='#008000')        
        sns.lineplot(x=x_single, y=y_avg[i], ax=ax, c='red')
        sns.lineplot(x=x_single, y=original[i], ax=ax, c='black')
        fig.savefig(f'{name}_{i}.pdf')
        plt.close()