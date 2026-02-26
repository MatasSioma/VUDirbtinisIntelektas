import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Klase 0: centras (2, 2), Klase 1: centras (6, 6)
class_0 = np.random.randn(15, 2) * 0.8 + np.array([2, 2])
class_1 = np.random.randn(15, 2) * 0.8 + np.array([6, 6])

X = np.concatenate((class_0, class_1))  # (30, 2)
y = np.array([0] * 15 + [1] * 15)

# ChatGpt ---
plt.figure(figsize=(8, 8))
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Klase 0', zorder=5)
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Klase 1', zorder=5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Sugeneruoti taškai')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('1Lab/1_duomenys.png', dpi=150, bbox_inches='tight')
plt.close()
# ----

print("~~~Sugeneruoti duomenys~~~`")
print("Klase 0")
print(f"Ilgis: {len(class_0)}")
print(" ".join(f"({str(x).replace('.', ',')}; {str(y).replace('.', ',')})" for x, y in class_0))
print("\nKlase 1")
print(f"Ilgis: {len(class_1)}")
print(" ".join(f"({str(x).replace('.', ',')}; {str(y).replace('.', ',')})" for x, y in class_1))