import os
import pandas as pd

interpolants = ['Linear', 'Trig', 'EncDec']
gammas = ['Trig', 'Sqrt', 'Quad']
couplings = ['Fourier', 'Quad']
data = ['2spirals', '8gaussians', 'checkerboard', 'circles', 'cos', 'line', 'moons', 'pinwheel', 'rings', 'swissroll']

df = pd.DataFrame(columns=('Mode', 'Interpolant', 'Gamma', 'Coupling', 'Dataset', 'Seed', 'Inference Time'))

for dt in data:
    for gamma in gammas:
        for interp in interpolants:
            for seed in range(3):
                name = f'results/SI/{dt}/{interp}_{gamma}/{seed}_1.0'
                with open(f'{name}/speed.txt', 'r') as f:
                        time = f.read().split(': ')[-1]
                df.loc[-1] = ['SI', interp, gamma, 'None', dt, seed, float(time)]
                df.index = df.index + 1
                for coupl in couplings:
                    # name = f'results/SI_Static/{dt}/{interp}_{gamma}/{coupl}/{seed}_1.0'
                    # with open(f'{name}/speed.txt', 'r') as f:
                    #     time = f.read().split(': ')[-1]
                    # df.loc[-1] = ['SI Static', interp, gamma, coupl, dt, seed, float(time)]
                    # df.index = df.index + 1

                    name = f'results/SB/{dt}/{interp}_{gamma}/{coupl}/{seed}_1.0'
                    with open(f'{name}/speed.txt', 'r') as f:
                        time = f.read().split(': ')[-1]
                    df.loc[-1] = ['SB', interp, gamma, coupl, dt, seed, float(time)]
                    df.index = df.index + 1

dataframe = df.groupby(["Dataset", "Mode"])['Inference Time'].mean()
print(dataframe)
# print(df)
