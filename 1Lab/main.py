import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Klase 0: centras (2, 2), Klase 1: centras (6, 6)
class_0 = np.random.randn(15, 2) * 0.8 + np.array([2, 2])
class_1 = np.random.randn(15, 2) * 0.8 + np.array([6, 6])

X = np.concatenate((class_0, class_1))  # (30, 2)
y = np.array([0] * 15 + [1] * 15)

# ChatGpt ---
PLOT_XLIM = (X[:, 0].min() - 1, X[:, 0].max() + 1)
PLOT_YLIM = (X[:, 1].min() - 1, X[:, 1].max() + 1)

def print_duomenys():
    plt.figure(figsize=(8, 8))
    plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Klase 0', zorder=5)
    plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Klase 1', zorder=5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Sugeneruoti taškai')
    plt.legend()
    plt.grid(True)
    plt.xlim(PLOT_XLIM)
    plt.ylim(PLOT_YLIM)
    plt.axis('equal')
    plt.savefig('1Lab/duomenys.png', dpi=150, bbox_inches='tight')
    plt.close()
# ----

    print("~~~Sugeneruoti duomenys~~~`")
    print("Klase 0")
    print(f"Ilgis: {len(class_0)}")
    print(" ".join(f"({str(x).replace('.', ',')}; {str(y).replace('.', ',')})" for x, y in class_0))
    print("\nKlase 1")
    print(f"Ilgis: {len(class_1)}")
    print(" ".join(f"({str(x).replace('.', ',')}; {str(y).replace('.', ',')})" for x, y in class_1))

def neuronas(x1, x2, w1, w2, b, fx):
    a = x1 * w1 + x2 * w2 + b
    return fx(a)

def slenkstis(a):
    return 1 if a >= 0 else 0

def sigmoid(a):
    return round(1.0 / (1.0 + np.exp(-a)))

AK_FUNK = slenkstis

def check(w1, w2, b):
    for i in range(len(X)):
        prediction = neuronas(X[i, 0], X[i, 1], w1, w2, b, AK_FUNK)
        if prediction != y[i]:
            return False
    return True

def find_weights():
    np.random.seed(0)
    weights = []  #(w1, w2, b)

    while len(weights) < 3:
        #siūlomi intervalai chatgpt
        w1 = np.random.uniform(-10, 10)
        w2 = np.random.uniform(-10, 10)
        b = np.random.uniform(-50, 50)

        if check(w1, w2, b):
            weights.append((w1, w2, b))
            print(f"Rastas rinkinys: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")

    return weights

def print_tieses(weights):
    spalvos = ['green', 'orange', 'purple']

    plt.figure(figsize=(8, 8))
    plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Klase 0', zorder=5)
    plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Klase 1', zorder=5)

    x_vals = np.linspace(PLOT_XLIM[0] + 1, PLOT_XLIM[1] -1, 200)

    for i, (w1, w2, b) in enumerate(weights):
        # Tiesė: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
        y_vals = -(w1 * x_vals + b) / w2
        plt.plot(x_vals, y_vals, color=spalvos[i],
                 label=f'Tiesė {i+1}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}')

        x_mid = (PLOT_XLIM[0] + PLOT_XLIM[1]) / 2
        y_mid = -(w1 * x_mid + b) / w2

        # chatgpt ---
        # Normalize vector for Visualization
        norm = np.sqrt(w1**2 + w2**2)
        scale = 1.5
        plt.annotate('', xy=(x_mid + scale * w1 / norm, y_mid + scale * w2 / norm),
                      xytext=(x_mid, y_mid),
                      arrowprops=dict(arrowstyle='->', color=spalvos[i], lw=2))
        #----

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Tiesės')
    plt.legend()
    plt.grid(True)
    plt.xlim(PLOT_XLIM)
    plt.ylim(PLOT_YLIM)
    plt.axis('equal')
    plt.savefig('1Lab/tieses.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    weights = find_weights()
    print_tieses(weights)