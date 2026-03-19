import numpy as np
import matplotlib.pyplot as plt
import time
import os

np.random.seed(42)  # Fiksuojame atsitiktinuma

DATA = os.path.join(os.path.dirname(__file__), 'breast-cancer-wisconsin.data')
REZULTATAI = os.path.join(os.path.dirname(__file__), 'rezultatai')

# Duomenu nuskaitymas ir paruosimas

def nuskaityti_duomenis(kelias):
    """Nuskaito CSV duomenis, pasalina trukstamas reiksmes ir ID stulpeli."""
    eilutes = []
    with open(kelias, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '?' in line:
                continue  # Praleisti eilutes su trukstamomis reiksmemis
            reiksmes = line.split(',')
            eilutes.append([int(r) for r in reiksmes])
    duomenys = np.array(eilutes)
    X = duomenys[:, 1:10].astype(float)  # 9 pozymiai (be ID)
    y = duomenys[:, 10]                  # Klases: 2 arba 4
    # Klases pakeitimas: 2 -> 0 , 4 -> 1
    y = np.where(y == 2, 0, 1).astype(float)
    return X, y

def normalizuoti(X_train, X_val, X_test):
    """Min-max normalizacija pagal mokymo duomenis."""
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    skirtumas = x_max - x_min
    skirtumas[skirtumas == 0] = 1  # Apsauga nuo dalybos is nulio
    X_train_n = (X_train - x_min) / skirtumas
    X_val_n = (X_val - x_min) / skirtumas
    X_test_n = (X_test - x_min) / skirtumas
    return X_train_n, X_val_n, X_test_n

def padalinti_duomenis(X, y, train_ratio=0.8, val_ratio=0.1):
    """Permaiso ir padalina duomenis i mokymo, validavimo ir testavimo aibes (80:10:10)."""
    indeksai = np.random.permutation(len(X))
    X = X[indeksai]
    y = y[indeksai]
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])

# Sigmoidinis neuronas

def sigmoid(a):
    """Sigmoidine aktyvacijos funkcija."""
    return 1.0 / (1.0 + np.exp(-a))

# Galutinis paklaidos ivertinimas (skaidre 48)

def ivertinti_paklaida(X, y_true, w, b):
    """
    totalError = 0
    FOR i = 1,2,...,m:
        a_i = 0
        FOR k = 0,1,...,n:
            a_i = a_i + w_k * x_ik
        y_i = f(a_i)
        error_i = (t_i - y_i)^2
        totalError = totalError + error_i
    totalError = totalError / m
    """
    m = len(X)
    n = len(w)
    totalError = 0
    for i in range(m):
        a_i = 0
        for k in range(n):
            a_i = a_i + w[k] * X[i, k]
        a_i = a_i + b  # Poslinkis (w_0 = b, x_0 = 1)
        y_i = sigmoid(a_i)
        error_i = (y_true[i] - y_i) ** 2
        totalError = totalError + error_i
    totalError = totalError / m
    return totalError

def ivertinti_tiksluma(X, y_true, w, b):
    """
    Klasifikavimo tikslumas: kiekvienam irasui apskaiciuojame klase,
    suapalinant sigmoidinio neurono isejima iki 0 arba 1.
    """
    m = len(X)
    n = len(w)
    teisingi = 0
    for i in range(m):
        a_i = 0
        for k in range(n):
            a_i = a_i + w[k] * X[i, k]
        a_i = a_i + b
        y_i = sigmoid(a_i)
        spejama_klase = round(y_i)  # Suapvaliname iki 0 arba 1
        if spejama_klase == int(y_true[i]):
            teisingi += 1
    return teisingi / m

# 4. Mokymo algoritmai

def stochastinis(X_train, y_train, X_val, y_val, eta, max_epochos, E_min, w_pradz, b_pradz):
    """
    totalError = inf, epoch = 0
    WHILE (totalError > E_min AND epoch < epochs):
        Sumaisyti mokymo duomenu irasus
        totalError = 0
        FOR i = 1,2,...,m:
            FOR k = 0,1,...,n:
                w_k := w_k - eta * (y_i - t_i) * y_i * (1 - y_i) * x_ik
            error = (t_i - y_i)^2
            totalError = totalError + error
        epoch = epoch + 1
    """
    w = w_pradz.copy()
    b = b_pradz
    m = len(X_train)
    n = len(w)  # Pozymiu skaicius

    # Rezultatu saugojimas
    train_mse_hist = []
    val_mse_hist = []
    train_acc_hist = []
    val_acc_hist = []

    totalError = float('inf')
    epoch = 0                  

    # (totalError > E_min AND epoch < epochs):
    while totalError > E_min and epoch < max_epochos:
        # Sumaisyti mokymo duomenu irasus
        indeksai = np.random.permutation(m)

        totalError = 0

        # FOR i = 1,2,...,m:
        for i in indeksai:
            xi = X_train[i]        # Vienas duomenu irasas
            ti = y_train[i]        # Tikroji klase (t_i)

            # Apskaiciuojame neurono isejima y_i
            a_i = 0
            for k in range(n):
                a_i = a_i + w[k] * xi[k]
            a_i = a_i + b
            yi = sigmoid(a_i)

            # Svoriu atnaujinimas (sigmoidinio neurono atvejis):
            # w_k := w_k - eta * (y_i - t_i) * y_i * (1 - y_i) * x_ik
            for k in range(n):
                w[k] = w[k] - eta * (yi - ti) * yi * (1 - yi) * xi[k]
            b = b - eta * (yi - ti) * yi * (1 - yi)  # Poslinkio atnaujinimas

            # error = (t_i - y_i)^2
            error = (ti - yi) ** 2
            # totalError = totalError + error
            totalError = totalError + error

        # epoch = epoch + 1
        epoch = epoch + 1

        # Epochos pabaigoje skaiciuojame paklaidas ir tiksluma (mokymo ir validavimo)
        train_mse = totalError / m  # Vidutine paklaida mokymo duomenims
        val_mse = ivertinti_paklaida(X_val, y_val, w, b)
        train_acc = ivertinti_tiksluma(X_train, y_train, w, b)
        val_acc = ivertinti_tiksluma(X_val, y_val, w, b)

        train_mse_hist.append(train_mse)
        val_mse_hist.append(val_mse)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

    return w, b, train_mse_hist, val_mse_hist, train_acc_hist, val_acc_hist

def paketinis(X_train, y_train, X_val, y_val, eta, max_epochos, E_min, w_pradz, b_pradz):
    """
    totalError = inf, epoch = 0
    WHILE (totalError > E_min AND epoch < epochs):
        Sumaisyti mokymo duomenu irasus
        totalError = 0
        gradientSum = (0,...,0)
        FOR i = 1,2,...,m:
            FOR k = 0,1,...,n:
                gradientSum_k = gradientSum_k + (y_i - t_i) * y_i * (1 - y_i) * x_ik
            error = (t_i - y_i)^2
            totalError = totalError + error
        FOR k = 0,1,...,n:
            w_k := w_k - eta * (gradientSum_k / m)
        epoch = epoch + 1
    """
    w = w_pradz.copy()
    b = b_pradz
    m = len(X_train)
    n = len(w)

    train_mse_hist = []
    val_mse_hist = []
    train_acc_hist = []
    val_acc_hist = []

    totalError = float('inf')
    epoch = 0                 

    # WHILE (totalError > E_min AND epoch < epochs):
    while totalError > E_min and epoch < max_epochos:
        # Sumaisyti mokymo duomenu irasus
        indeksai = np.random.permutation(m)

        totalError = 0                
        gradientSum = np.zeros(n)  
        gradientSum_b = 0.0     

        # FOR i = 1,2,...,m:
        for idx in indeksai:
            xi = X_train[idx]
            ti = y_train[idx]

            # Apskaiciuojame neurono isejima y_i
            a_i = 0
            for k in range(n):
                a_i = a_i + w[k] * xi[k]
            a_i = a_i + b
            yi = sigmoid(a_i)

            # gradientSum_k = gradientSum_k + (y_i - t_i) * y_i * (1 - y_i) * x_ik
            for k in range(n):
                gradientSum[k] = gradientSum[k] + (yi - ti) * yi * (1 - yi) * xi[k]
            gradientSum_b = gradientSum_b + (yi - ti) * yi * (1 - yi)

            # error = (t_i - y_i)^2
            error = (ti - yi) ** 2
            # totalError = totalError + error
            totalError = totalError + error

        # FOR k = 0,1,...,n:
        #     w_k := w_k - eta * (gradientSum_k / m)
        for k in range(n):
            w[k] = w[k] - eta * (gradientSum[k] / m)
        b = b - eta * (gradientSum_b / m)

        # epoch = epoch + 1
        epoch = epoch + 1

        # Epochos pabaigoje skaiciuojame paklaidas ir tiksluma
        train_mse = totalError / m
        val_mse = ivertinti_paklaida(X_val, y_val, w, b)
        train_acc = ivertinti_tiksluma(X_train, y_train, w, b)
        val_acc = ivertinti_tiksluma(X_val, y_val, w, b)

        train_mse_hist.append(train_mse)
        val_mse_hist.append(val_mse)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

    return w, b, train_mse_hist, val_mse_hist, train_acc_hist, val_acc_hist

# Grafiku braižymas

def paklaidu_grafikas(train_mse, val_mse, pavadinimas, failo_vardas):
    """Paklaidos priklausomybe nuo epochu (mokymo ir validavimo)."""
# CHATGPT ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_mse) + 1), train_mse, label='Mokymo paklaida', color='blue')
    plt.plot(range(1, len(val_mse) + 1), val_mse, label='Validavimo paklaida', color='red')
    plt.xlabel('Epocha')
    plt.ylabel('Vidutinė kvadratinė paklaida')
    plt.title(pavadinimas)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REZULTATAI, failo_vardas), dpi=150, bbox_inches='tight')
    plt.close()

def tikslumo_grafikas(train_acc, val_acc, pavadinimas, failo_vardas):
    """Klasifikavimo tikslumas priklausomybe nuo epochu."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Mokymo tikslumas', color='blue')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validavimo tikslumas', color='red')
    plt.xlabel('Epocha')
    plt.ylabel('Tikslumas')
    plt.title(pavadinimas)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REZULTATAI, failo_vardas), dpi=150, bbox_inches='tight')
    plt.close()
# ---

def lr_palyginimas(rezultatai, metodo_pavadinimas, failo_vardas):
    """Stulpeline diagrama: skirtingu mokymosi greicio itaka tikslumui ir paklaidai."""
    lr_values = [r['lr'] for r in rezultatai]
    val_acc = [r['val_acc'] for r in rezultatai]
    val_mse = [r['val_mse'] for r in rezultatai]

    x = np.arange(len(lr_values))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(x - width/2, val_acc, width, label='Validavimo tikslumas', color='steelblue')
    ax1.set_ylabel('Tikslumas')
    ax1.set_xlabel('Mokymosi greitis (η)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(lr) for lr in lr_values])
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, val_mse, width, label='Validavimo paklaida', color='salmon')
    ax2.set_ylabel('MSE paklaida')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_title(f'{metodo_pavadinimas}: mokymosi greičio įtaka')
    fig.tight_layout()
    plt.savefig(os.path.join(REZULTATAI, failo_vardas), dpi=150, bbox_inches='tight')
    plt.close()

def metodu_palyginimas(sgd_rez, bgd_rez, failo_vardas):
    """Stulpeline diagrama: Stochastinio ir paketinio nusileidimo palyginimas (tikslumas, paklaida, laikas)."""
    kategorijos = ['Val. tikslumas', 'Test tikslumas', 'Val. paklaida', 'Test paklaida']
    sgd_vals = [sgd_rez['val_acc'], sgd_rez['test_acc'], sgd_rez['val_mse'], sgd_rez['test_mse']]
    bgd_vals = [bgd_rez['val_acc'], bgd_rez['test_acc'], bgd_rez['val_mse'], bgd_rez['test_mse']]

    x = np.arange(len(kategorijos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, sgd_vals, width, label='Stochastinis gradientinis nusileidimas', color='steelblue')
    ax.bar(x + width/2, bgd_vals, width, label='Paketinis gradientinis nusileidimas', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(kategorijos)
    ax.legend()
    ax.set_title('Stochastinis ir Paketinis gradientinis nusileidimas')
    ax.grid(True, axis='y')
    plt.savefig(os.path.join(REZULTATAI, failo_vardas), dpi=150, bbox_inches='tight')
    plt.close()

# CHATGPT
def laiko_palyginimas(sgd_laikas, bgd_laikas, failo_vardas):
    """Stulpeline diagrama: mokymo laiko palyginimas."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(['Stochastinis GD', 'Paketinis GD'], [sgd_laikas, bgd_laikas],
           color=['steelblue', 'salmon'])
    ax.set_ylabel('Laikas (s)')
    ax.set_title('Mokymo laiko palyginimas (tos pačios epochos)')
    ax.grid(True, axis='y')
    for i, v in enumerate([sgd_laikas, bgd_laikas]):
        ax.text(i, v + 0.001, f'{v:.4f}s', ha='center')
    plt.savefig(os.path.join(REZULTATAI, failo_vardas), dpi=150, bbox_inches='tight')
    plt.close()
# ---

# Testavimas

def testuoti(X_test, y_test, w, b):
    """
    Grazina paklaida, tiksluma ir spejamas klases kiekvienam irasui.
    """
    mse = ivertinti_paklaida(X_test, y_test, w, b)
    acc = ivertinti_tiksluma(X_test, y_test, w, b)
    # Spejamos klases kiekvienam irasui
    n = len(w)
    klases_spejamos = []
    for i in range(len(X_test)):
        a_i = 0
        for k in range(n):
            a_i = a_i + w[k] * X_test[i, k]
        a_i = a_i + b
        y_i = sigmoid(a_i)
        klases_spejamos.append(round(y_i))
    return mse, acc, klases_spejamos


# CHATGPT
def spausdinti_rezultatus(pavadinimas, w, b, train_mse, val_mse, train_acc, val_acc,
                          test_mse, test_acc, klases_spejamos, y_test, laikas):
    """Isvedame visus rezultatus i konsole."""
    print(f"\n{'='*60}")
    print(f"  {pavadinimas}")
    print(f"{'='*60}")

    print(f"\nGauti svoriai:")
    for k in range(len(w)):
        print(f"  w{k+1} = {w[k]:.6f}")
    print(f"  b (poslinkis) = {b:.6f}")

    print(f"\nEpochu skaicius: {len(train_mse)}")
    print(f"Mokymo laikas: {laikas:.4f} s")

    print(f"\nPaskutines epochos rezultatai:")
    print(f"  Mokymo paklaida:     {train_mse[-1]:.6f}")
    print(f"  Validavimo paklaida: {val_mse[-1]:.6f}")
    print(f"  Mokymo tikslumas:    {train_acc[-1]:.4f} ({train_acc[-1]*100:.2f}%)")
    print(f"  Validavimo tikslumas:{val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)")

    print(f"\nTestavimo rezultatai:")
    print(f"  Paklaida: {test_mse:.6f}")
    print(f"  Tikslumas: {test_acc:.4f} ({test_acc*100:.2f}%)")

    print(f"\nTestavimo duomenu klasifikacija:")
    print(f"  {'Nr.':<5} {'Spejama':<10} {'Tikroji':<10} {'Teisingai?'}")
    print(f"  {'-'*40}")
    for i in range(len(y_test)):
        teisinga = 'Taip' if klases_spejamos[i] == int(y_test[i]) else 'Ne'
        print(f"  {i+1:<5} {klases_spejamos[i]:<10} {int(y_test[i]):<10} {teisinga}")
# ---

# PALEIDIMAS

if __name__ == "__main__":
    print("Nuskaitomi duomenys...")
    X, y = nuskaityti_duomenis(DATA)
    print(f"Duomenu irasu skaicius (be trukstamu): {len(X)}")
    print(f"Pozymiu skaicius: {X.shape[1]}")
    print(f"Klase 0 (nepiktybinis): {np.sum(y == 0):.0f}")
    print(f"Klase 1 (piktybinis):   {np.sum(y == 1):.0f}")

    # Padalijimas i mokymo, validavimo ir testavimo aibes
    X_train, y_train, X_val, y_val, X_test, y_test = padalinti_duomenis(X, y)
    print(f"\nDuomenu padalijimas (80:10:10):")
    print(f"  Mokymo:     {len(X_train)}")
    print(f"  Validavimo: {len(X_val)}")
    print(f"  Testavimo:  {len(X_test)}")

    # Normalizacija pagal mokymo duomenis
    X_train, X_val, X_test = normalizuoti(X_train, X_val, X_test)

    # --- Pradiniu svoriu inicializacija ---
    n_pozymiu = X_train.shape[1]  # 9 pozymiai
    w_pradz = np.random.uniform(-0.5, 0.5, size=n_pozymiu)
    b_pradz = np.random.uniform(-0.5, 0.5)
    print(f"\nPradiniai svoriai: {w_pradz}")
    print(f"Pradinis poslinkis: {b_pradz:.4f}")

    # --- Hiperparametrai ---
    MAX_EPOCHOS = 150
    ETA = 0.75       # Pagrindinis mokymosi greitis
    E_MIN = 0.01    # Minimali paklaida sustojimui

    # A) Stochastinis gradientinis nusileidimas
    print("\n--- Stochastinis mokymas ---")
    start = time.time()
    sgd_w, sgd_b, sgd_train_mse, sgd_val_mse, sgd_train_acc, sgd_val_acc = \
        stochastinis(X_train, y_train, X_val, y_val, ETA, MAX_EPOCHOS, E_MIN, w_pradz, b_pradz)
    sgd_laikas = time.time() - start

    sgd_test_mse, sgd_test_acc, sgd_test_klases = testuoti(X_test, y_test, sgd_w, sgd_b)
    spausdinti_rezultatus("STOCHASTINIS GRADIENTINIS NUSILEIDIMAS",
                          sgd_w, sgd_b, sgd_train_mse, sgd_val_mse,
                          sgd_train_acc, sgd_val_acc,
                          sgd_test_mse, sgd_test_acc, sgd_test_klases, y_test, sgd_laikas)

    # Grafikai
    paklaidu_grafikas(sgd_train_mse, sgd_val_mse,
        'Stochastinis gradientinis nusileidimas: paklaida nuo epochos', 'sgd_paklaida.png')
    tikslumo_grafikas(sgd_train_acc, sgd_val_acc,
        'Stochastinis gradientinis nusileidimas: tikslumas nuo epochos', 'sgd_tikslumas.png')

    # B) Paketinis gradientinis nusileidimas
    print("\n--- Paketinis mokymas ---")
    start = time.time()
    bgd_w, bgd_b, bgd_train_mse, bgd_val_mse, bgd_train_acc, bgd_val_acc = \
        paketinis(X_train, y_train, X_val, y_val, ETA, MAX_EPOCHOS, E_MIN, w_pradz, b_pradz)
    bgd_laikas = time.time() - start

    bgd_test_mse, bgd_test_acc, bgd_test_klases = testuoti(X_test, y_test, bgd_w, bgd_b)
    spausdinti_rezultatus("PAKETINIS GRADIENTINIS NUSILEIDIMAS",
                          bgd_w, bgd_b, bgd_train_mse, bgd_val_mse,
                          bgd_train_acc, bgd_val_acc,
                          bgd_test_mse, bgd_test_acc, bgd_test_klases, y_test, bgd_laikas)

    # Grafikai
    paklaidu_grafikas(bgd_train_mse, bgd_val_mse,
        'Paketinis gradientinis nusileidimas: paklaida nuo epochos', 'bgd_paklaida.png')
    tikslumo_grafikas(bgd_train_acc, bgd_val_acc,
        'Paketinis gradientinis nusileidimas: tikslumas nuo epochos', 'bgd_tikslumas.png')

    # C) Mokymosi greicio tyrimas (3 skirtingos reiksmes)
    print(f"\n\n{'='*60}")
    print("  MOKYMOSI GREICIO TYRIMAS")
    print('='*60)

    lr_values = [ 0.5, 1, 2.5, 5, 10]
    sgd_lr_rezultatai = []
    bgd_lr_rezultatai = []

    for lr in lr_values:
        print(f"\n--- η = {lr} ---")

        # Stochastinis su skirtigu lr
        _, _, s_train_mse, s_val_mse, s_train_acc, s_val_acc = \
            stochastinis(X_train, y_train, X_val, y_val, lr, MAX_EPOCHOS, E_MIN, w_pradz, b_pradz)
        sgd_lr_rezultatai.append({
            'lr': lr,
            'val_acc': s_val_acc[-1],
            'val_mse': s_val_mse[-1],
            'train_acc': s_train_acc[-1],
            'train_mse': s_train_mse[-1]
        })
        print(f"  SGD: val_acc={s_val_acc[-1]:.4f}, val_mse={s_val_mse[-1]:.6f}")

        # Paketinis su skirtigu lr
        _, _, b_train_mse, b_val_mse, b_train_acc, b_val_acc = \
            paketinis(X_train, y_train, X_val, y_val, lr, MAX_EPOCHOS, E_MIN, w_pradz, b_pradz)
        bgd_lr_rezultatai.append({
            'lr': lr,
            'val_acc': b_val_acc[-1],
            'val_mse': b_val_mse[-1],
            'train_acc': b_train_acc[-1],
            'train_mse': b_train_mse[-1]
        })
        print(f"  BGD: val_acc={b_val_acc[-1]:.4f}, val_mse={b_val_mse[-1]:.6f}")

    # Grafikai: mokymosi greicio itaka
    lr_palyginimas(sgd_lr_rezultatai, 'Stochastinis', 'sgd_lr_palyginimas.png')
    lr_palyginimas(bgd_lr_rezultatai, 'Paketinis', 'bgd_lr_palyginimas.png')

    # D) Metodu palyginimas (Sigmoidinis ir paketinis)
    sgd_rez = {
        'val_acc': sgd_val_acc[-1], 'test_acc': sgd_test_acc,
        'val_mse': sgd_val_mse[-1], 'test_mse': sgd_test_mse
    }
    bgd_rez = {
        'val_acc': bgd_val_acc[-1], 'test_acc': bgd_test_acc,
        'val_mse': bgd_val_mse[-1], 'test_mse': bgd_test_mse
    }
    metodu_palyginimas(sgd_rez, bgd_rez, 'metodu_palyginimas.png')
    laiko_palyginimas(sgd_laikas, bgd_laikas, 'laiko_palyginimas.png')

    print(f"\n\nMokymo laiko palyginimas ({MAX_EPOCHOS} max epochu):")
    print(f"  Stochastinis: {sgd_laikas:.4f} s")
    print(f"  Paketinis:    {bgd_laikas:.4f} s")