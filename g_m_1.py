import numpy as np

from fractions import Fraction

def format_fraction(value, max_denominator=1000):
    """
    Convert a float to a fraction string representation.
    
    Args:
        value: Number to convert
        max_denominator: Maximum denominator to allow in fraction
    
    Returns:
        str: Fraction representation or decimal if conversion not practical
    """
    if isinstance(value, (int, float)):
        if value.is_integer():
            return str(int(value))
        try:
            f = Fraction(value).limit_denominator(max_denominator)
            if f.denominator == 1:
                return str(f.numerator)
            return f"{f.numerator}/{f.denominator}"
        except:
            return f"{value:.8f}"
    return str(value)

def equation_sigma(sigma, mu, fn_laplace, *args):
    """
    Équation à résoudre : σ = A*(μ(1-σ))
    où A* est la transformée de Laplace de la distribution du temps d'arrivée
    
    Args:
        sigma: Variable à résoudre
        mu: Taux de service
        fn_laplace: Fonction de transformée de Laplace
        *args: Arguments pour la fonction de Laplace
    """
    return sigma - fn_laplace(mu * (1 - sigma), *args)

def newton_raphson(f, x0, args=(), tol=1e-6, max_iter=100):
    """
    Méthode de Newton-Raphson pour trouver les racines
    
    Args:
        f: Fonction dont on cherche la racine
        x0: Estimation initiale
        args: Arguments supplémentaires pour f
        tol: Tolérance pour la convergence
        max_iter: Nombre maximum d'itérations
        
    Returns:
        float: Racine de l'équation
        
    Raises:
        RuntimeError: Si la méthode ne converge pas
    """
    x0 = float(x0)  # Assurer que x0 est un float pour la stabilité numérique
    
    for _ in range(max_iter):
        fx = f(x0, *args)
        # Calculer la dérivée numérique avec différence centrale
        h = max(tol, abs(x0 * 1e-8))  # Pas adaptatif
        fpx = (f(x0 + h, *args) - f(x0 - h, *args)) / (2 * h)
        
        if abs(fpx) < tol:
            raise RuntimeError("Dérivée trop proche de zéro")
            
        x1 = x0 - fx / fpx
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
        
    raise RuntimeError(f"Échec de convergence après {max_iter} itérations")

def calculer_performance(sigma, mu, lambda_):
    """
    Calculer les métriques de performance de la file d'attente en utilisant σ
    
    Args:
        sigma: Solution de l'équation fonctionnelle
        mu: Taux de service
        lambda_: Taux d'arrivée effectif
        
    Returns:
        tuple: (Q, R, W) où:
            Q: Nombre moyen de clients dans le système
            R: Temps de réponse moyen
            W: Temps d'attente moyen
    """
    R = 1/mu * (1/(1 - sigma))  # Temps de réponse moyen
    Q = lambda_ * R             # Nombre moyen dans le système
    W = R - 1/mu               # Temps d'attente moyen
    return Q, R, W

def obtenir_lambda(probs, lambdas):
    """
    Calculer le taux d'arrivée effectif à partir des paramètres hyper-exponentiels
    
    Args:
        probs: Liste des probabilités
        lambdas: Liste des paramètres de taux
        
    Returns:
        float: Taux d'arrivée effectif
    """
    temps_moyen = sum(p * (1/l) for p, l in zip(probs, lambdas))
    return 1/temps_moyen

def laplace_hyperexp(s, probs, lambdas):
    """
    Transformée de Laplace de la distribution hyper-exponentielle (Hk)
    
    Args:
        s: Variable de la transformée de Laplace
        probs: Liste des probabilités pour chaque branche
        lambdas: Liste des paramètres de taux pour chaque branche
        
    Returns:
        float: Valeur de la transformée de Laplace en s
        
    Raises:
        ValueError: Si les probabilités ne somment pas à 1 ou si les longueurs ne correspondent pas
    """
    if len(probs) != len(lambdas):
        raise ValueError("Le nombre de probabilités doit correspondre au nombre de taux")
    if not np.isclose(sum(probs), 1.0, rtol=1e-5):
        raise ValueError("Les probabilités doivent sommer à 1")
    if any(p < 0 or p > 1 for p in probs):
        raise ValueError("Toutes les probabilités doivent être entre 0 et 1")
    if any(l <= 0 for l in lambdas):
        raise ValueError("Tous les taux doivent être positifs")
        
    return sum(p * lambda_i / (lambda_i + s) 
              for p, lambda_i in zip(probs, lambdas))


def laplace_hypoexp(s, taux):
    """
    Transformée de Laplace de la distribution hypo-exponentielle (somme de k exponentielles)
    
    Args:
        s: Variable de la transformée de Laplace
        taux: Liste des taux pour chaque phase
        
    Returns:
        float: Valeur de la transformée de Laplace en s
        
    Raises:
        ValueError: Si les taux ne sont pas valides
    """
    if any(t <= 0 for t in taux):
        raise ValueError("Tous les taux doivent être positifs")
    
    # Pour la distribution hypo-exponentielle, la transformée de Laplace est le produit
    # des transformées de Laplace des distributions exponentielles individuelles
    resultat = 1.0
    for mu_i in taux:
        resultat *= (mu_i / (mu_i + s))
    return resultat

def laplace_exp(s, lambda_):
    """Transformée de Laplace de la distribution exponentielle"""
    if lambda_ <= 0:
        raise ValueError("Le taux doit être positif")
    return lambda_ / (lambda_ + s)

def laplace_erlang(s, k, mu):
    """Transformée de Laplace de la distribution Erlang-k"""
    if k <= 0 or not isinstance(k, int):
        raise ValueError("k doit être un entier positif")
    if mu <= 0:
        raise ValueError("mu doit être positif")
    return (mu / (mu + s)) ** k

def laplace_deterministe(s, D):
    if D <= 0:
        """Transformée de Laplace de la distribution déterministe"""
        raise ValueError("D doit être positif")
    return np.exp(-s * D)


def afficher_resultats(resultats, indentation=2):
    """
    Afficher les résultats d'analyse de file d'attente dans un format lisible,
    utilisant des fractions quand possible.
    
    Args:
        resultats: Dictionnaire retourné par file_gm1_generale
        indentation: Nombre d'espaces pour l'indentation
    """
    espace = ' ' * indentation
    
    if "erreur" in resultats:
        print(f"{espace}Erreur: {resultats['erreur']}")
        return
    
    nom_dist = resultats.get('distribution', 'inconnue').capitalize()
    print(f"{espace}Résultats pour la Distribution {nom_dist}:")
    
    metriques = [
        ('Taux d\'Arrivée Effectif (λ)', 'taux_arrivee_effectif'),
        ('Intensité du Trafic (ρ, Utilisation U)', 'intensite_trafic'),
        ('Sigma (σ)', 'sigma'),
        ('Clients Moyens dans le Système', 'clients_moyens'),
        ('Temps de Réponse Moyen', 'temps_reponse_moyen'),
        ('Temps d\'Attente Moyen', 'temps_attente_moyen')
    ]
    
    for nom, cle in metriques:
        if cle in resultats:
            valeur = format_fraction(resultats[cle])
            print(f"{espace}- {nom}: {valeur}")
    
    print("-" * 50)

def file_gm1_generale(mu, distribution='hyperexp', fn_laplace=None, 
                     probs=None,  lambdas=None, **args_dist):
    """
    [Previous documentation remains the same]
    """
    try:

        if distribution == 'hyperexp':
            if probs is None or lambdas is None:
                raise ValueError("probs et lambdas requis pour hyperexp")
                
            if len(probs) != len(lambdas):
                raise ValueError("probs et lambdas doivent avoir la même longueur")
            if not np.isclose(sum(probs), 1.0, rtol=1e-5):
                raise ValueError("probs doivent sommer à 1")
            
            fn_laplace = laplace_hyperexp
            args_laplace = (probs, lambdas)

        elif distribution == 'hypoexp':
            taux_hypoexp = args_dist.get('taux_hypoexp')
            if taux_hypoexp is None or len(taux_hypoexp) < 2:
                raise ValueError("Au moins deux taux sont requis pour hypoexp")
            
            fn_laplace = laplace_hypoexp
            args_laplace = (taux_hypoexp,)
            
        elif distribution == 'erlang':
            k = args_dist.get('k', 1)
            fn_laplace = laplace_erlang
            args_laplace = (k, args_dist['mu'])
            
        else:
            raise ValueError(f"Distribution inconnue: {distribution}")

        # Calculer le taux d'arrivée effectif
        h = 1e-6
        A0 = fn_laplace(0, *args_laplace)
        Ah = fn_laplace(h, *args_laplace)
        E_T = (A0 - Ah)/h
        lambda_ = 1/E_T
        
        # Vérifier la stabilité
        rho = lambda_/mu
        if rho >= 1:
            return {"erreur": f"Instable (ρ = {format_fraction(rho)} ≥ 1)"}

        # Résoudre pour σ
        sigma = newton_raphson(
            equation_sigma, 
            0.5,
            args=(mu, fn_laplace) + args_laplace
        )

        # Calculer les métriques de performance
        Q, R, W = calculer_performance(sigma, mu, lambda_)
        
        return {
            "distribution": distribution,
            "sigma": sigma,
            "intensite_trafic": rho,
            "clients_moyens": Q,
            "temps_reponse_moyen": R,
            "temps_attente_moyen": W,
            "taux_arrivee_effectif": lambda_
        }
        
    except Exception as e:
        return {"erreur": str(e)}

if __name__ == "__main__":
    print("Exemple Hyper-exponentiel:")
    res = file_gm1_generale(
        mu=5.0,
        # for hypo 
        distribution='erlang',
        # taux_hypoexp=[10, 5]

        # for hyper
        # distribution='hyperexp',
        # probs=[1, 1],
        # lambdas=[10, 5]

        # for erlang k
        k = 2
    )
    afficher_resultats(res)

