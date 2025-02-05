import numpy as np

def mva_algorithm(M, N, mu, e):
    """
    Implémentation de l'algorithme MVA avec calcul des débits par station
    
    Paramètres:
    M : nombre de stations
    N : nombre de clients dans le système
    mu : tableau des taux de service (M)
    e : tableau des visites (M)
    
    Retourne:
    Q : longueurs moyennes des files
    X : débit global
    X_i : débits par station
    R : temps de réponse
    """
    
    # Initialisation
    Q = np.zeros(M)  # Files d'attente moyennes
    R = np.zeros(M)  # Temps de réponse
    X = 0  # Débit global
    X_i = np.zeros(M)  # Débits par station
    
    # Pour chaque population possible de 1 à N
    for n in range(1, N + 1):
        # Calculer le temps de réponse pour chaque station
        for i in range(M):
            R[i] = (1 / mu[i]) * (1 + Q[i])
        
        # Calculer le débit global
        X = n / np.sum(e * R)
        
        # Calculer les débits par station
        X_i = X * e
        
        # Mettre à jour les longueurs moyennes des files
        for i in range(M):
            Q[i] = X * e[i] * R[i]
    
    return Q, X, X_i, R

def exemple_utilisation():
    """
    Exemple d'utilisation de l'algorithme MVA
    """
    print("Exemple d'utilisation de l'algorithme MVA:")
    
    # Paramètres
    M = 3  # Nombre de stations
    N = 1  # Nombre de clients
    
    # Taux de service
    mu = np.array([
        1,
        4/15,
        1/5
    ])
    
    # Nombre de visites
    e = np.array([
        1,
        8/15,
        4/5
    ])
    
    # Exécution de l'algorithme
    Q, X, X_i, R = mva_algorithm(M, N, mu, e)
    
    # Affichage des résultats
    print("\nRésultats pour", N, "clients:")
    print("\nLongueurs moyennes des files (Q):")
    for i, q in enumerate(Q):
        print(f"Station {i+1}: {q:.4f}")
    
    print("\nDébit global du système (X):")
    print(f"X = {X:.4f}")
    
    print("\nDébits par station (X_i):")
    for i, x in enumerate(X_i):
        print(f"Station {i+1}: {x:.4f}")
    
    print("\nTemps de réponse (R):")
    for i, r in enumerate(R):
        print(f"Station {i+1}: {r:.4f}")
    
    print("\nTemps de réponse total:")
    print(f"R total = {np.sum(e * R):.4f}")
    
    print("\nNombre moyen de clients dans le système:")
    print(f"N moyen = {np.sum(Q):.4f}")

if __name__ == "__main__":
    exemple_utilisation()
