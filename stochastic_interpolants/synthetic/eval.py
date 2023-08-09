import os

interpolants = ['Linear', 'Trig', 'EncDec']
gammas = ['Trig', 'Sqrt', 'Quad']
couplings = ['Fourier', 'Quad']
data = ['2spirals', '8gaussians', 'checkerboard', 'circles', 'cos', 'line', 'moons', 'pinwheel', 'rings', 'swissroll']

f = 'mmd.txt'
i = 0
j = 0
k = 0

for dt in data:
    for gamma in gammas:
        for interp in interpolants:
            for seed in range(3):
                name = f'results/SI/{dt}/{interp}_{gamma}/{seed}_1.0'
                if not os.path.exists(f'{name}/{f}'):
                    print(name)
                    i += 1
                for coupl in couplings:
                    name = f'results/SI_Static/{dt}/{interp}_{gamma}/{coupl}/{seed}_1.0'
                    if not os.path.exists(f'{name}/{f}'):
                        print(name)
                        j += 1

                    name = f'results/SB/{dt}/{interp}_{gamma}/{coupl}/{seed}_1.0'
                    if not os.path.exists(f'{name}/{f}'):
                        print(name)
                        k += 1

print(i)
print(j)
print(k)