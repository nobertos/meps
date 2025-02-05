import numpy as np
from scipy.stats import expon


from fractions import Fraction

def format_fraction(value):
    """Helper function to format numbers as fractions when possible."""
    if isinstance(value, (int, float)):
        if value.is_integer():
            return str(int(value))
        try:
            f = Fraction(value).limit_denominator()
            if f.denominator == 1:
                return str(f.numerator)
            return f"{f.numerator}/{f.denominator}"
        except:
            return f"{value:.8f}"
    return str(value)

class FileAttenteMG1:
    """
    Analyse d'une file d'attente M/G/1 utilisant la formule de Pollaczek-Khinchin.
    Supporte les distributions:
    - Erlang-k (CV² < 1)
    - Hyper-exponential-k (CV² > 1)
    - Hypo-exponential-k (CV² < 1)
    - Exponentielle (CV² = 1)
    - Constante (CV² = 0)
    
    Notation:
    - m₁: Premier moment du temps de service (E[X])
    - m₂: Second moment du temps de service (E[X²])
    - CV²: Coefficient de variation au carré = (m₂ - m₁²)/m₁²
    """
    
    def __init__(self, taux_arrivee):
        if taux_arrivee <= 0:
            raise ValueError("Le taux d'arrivée doit être positif")
        self.lambda_ = taux_arrivee
    
    def calculer_cv_carre(self, m1, m2):
        """
        Calcule le coefficient de variation au carré en utilisant les moments.
        
        Args:
            m1: Premier moment (moyenne)
            m2: Second moment
            
        Returns:
            float: Coefficient de variation au carré
        """
        return (m2 - m1**2) / (m1**2)
    
    def analyser_avec_moments(self, m1, m2):
        """
        Analyse la file en utilisant les deux premiers moments de la distribution du temps de service.
        
        Args:
            m1: Premier moment E[X]
            m2: Second moment E[X²]
        """
        # Intensité du trafic
        rho = self.lambda_ * m1
        
        if rho >= 1:
            return {
                "erreur": f"Système instable : intensité du trafic ρ = {rho:.8f} ≥ 1",
                "intensite_trafic": rho
            }
        
        # Calcul du CV² en utilisant les moments
        cv_carre = self.calculer_cv_carre(m1, m2)
        
        # Calcul des métriques de performance avec la formule P-K
        EWq = (self.lambda_ * m2) / (2 * (1 - rho))
        ER = EWq + m1
        EQq = self.lambda_ * EWq
        EQ = self.lambda_ * ER
        
        return {
            "intensite_trafic": rho,
            "E[R]": ER,           # Temps de réponse moyen
            "E[Wq]": EWq,         # Temps d'attente moyen
            "E[Q]": EQ,           # Nombre moyen dans le système
            "E[Qq]": EQq,         # Longueur moyenne de la file
            "E[X]": m1,           # Premier moment (temps de service moyen)
            "E[X²]": m2,          # Second moment
            "CV²": cv_carre,      # Coefficient de variation au carré
            "CV": np.sqrt(cv_carre),  # Coefficient de variation
            "utilisation": rho
        }
    
    def analyser_service_erlang(self, EX, k):
        """
        Analyse la file avec des temps de service suivant une loi d'Erlang-k.
        
        Pour la distribution d'Erlang-k:
        - m₁ = EX
        - m₂ = EX² * (k+1)/k
        - CV² = 1/k
        
        Args:
            EX: Temps de service moyen (m₁)
            k: Paramètre de forme (nombre de phases)
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k doit être un entier positif")
        
        # Calcul des moments pour Erlang-k
        m1 = EX
        m2 = (EX**2) * ((k + 1) / k)
        
        resultats = self.analyser_avec_moments(m1, m2)
        
        if "erreur" not in resultats:
            resultats["distribution"] = f"Erlang-{k}"
            resultats["parametre_forme"] = k
            resultats["taux_phase"] = k/EX
            resultats["interpretation"] = {
                "phases": f"Le processus de service consiste en {k} phases exponentielles en série",
                "variabilite": "Faible" if k > 5 else "Moyenne" if k > 1 else "Élevée",
                "cv_carre_theorique": 1/k
            }
        
        return resultats
    
    def analyser_service_hyperexp(self, EX, k, p=None, mu=None):
        """
        Analyse avec temps de service Hyper-exponentiels (mélange de k distributions exponentielles).
        
        Args:
            EX: Temps de service moyen souhaité
            k: Nombre de branches exponentielles
            p: Liste des probabilités de chaque branche (optionnel)
            mu: Liste des taux de service de chaque branche (optionnel)
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k doit être un entier ≥ 2")
            
        # Si p n'est pas fourni, utiliser des probabilités égales
        if p is None:
            p = np.ones(k) / k
        elif len(p) != k or not np.isclose(sum(p), 1):
            raise ValueError("p doit être un vecteur de probabilités de somme 1")
            
        # Si mu n'est pas fourni, calculer des taux qui donnent EX
        if mu is None:
            # Utiliser des taux qui donnent le CV² > 1 souhaité
            mu = np.array([(2*i+1)/(EX) for i in range(k)])
        elif len(mu) != k:
            raise ValueError("mu doit avoir k éléments")
            
        # Calcul des moments
        m1 = sum(p[i]/mu[i] for i in range(k))
        m2 = 2 * sum(p[i]/mu[i]**2 for i in range(k))
        
        resultats = self.analyser_avec_moments(m1, m2)
        
        if "erreur" not in resultats:
            resultats["distribution"] = f"Hyper-exponential-{k}"
            resultats["parametres"] = {
                "probabilites": p.tolist(),
                "taux_service": mu.tolist()
            }
            resultats["interpretation"] = {
                "phases": f"Mélange de {k} distributions exponentielles en parallèle",
                "variabilite": "Très élevée",
                "cv_carre_theorique": self.calculer_cv_carre(m1, m2)
            }
        
        return resultats
    
    def analyser_service_hypoexp(self, EX, k, mu=None):
        """
        Analyse avec temps de service Hypo-exponentiels (k phases exponentielles en série avec taux différents).
        
        Args:
            EX: Temps de service moyen souhaité
            k: Nombre de phases
            mu: Liste des taux de service de chaque phase (optionnel)
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k doit être un entier ≥ 2")
            
        # Si mu n'est pas fourni, calculer des taux qui donnent EX
        if mu is None:
            # Utiliser des taux différents pour chaque phase
            mu = np.array([k/EX * (1 + 0.2*i) for i in range(k)])
        elif len(mu) != k:
            raise ValueError("mu doit avoir k éléments")
            
        # Calcul des moments (formule récursive)
        m1 = sum(1/mu[i] for i in range(k))
        
        # Calcul du second moment
        m2 = 2 * sum(
            sum(1/(mu[i]*mu[j]) for j in range(i+1))
            for i in range(k)
        )
        
        resultats = self.analyser_avec_moments(m1, m2)
        
        if "erreur" not in resultats:
            resultats["distribution"] = f"Hypo-exponential-{k}"
            resultats["parametres"] = {
                "taux_service": mu.tolist()
            }
            resultats["interpretation"] = {
                "phases": f"{k} phases exponentielles en série avec taux différents",
                "variabilite": "Faible à moyenne",
                "cv_carre_theorique": self.calculer_cv_carre(m1, m2)
            }
        
        return resultats

    def analyser_service_exponentiel(self, temps_service_moyen):
        """
        Cas spécial: file M/M/1 (temps de service exponentiels).
        Équivalent à des temps de service Erlang-1.
        """
        return self.analyser_service_erlang(temps_service_moyen, k=1)
    
    def analyser_service_constant(self, temps_service):
        """
        Cas spécial: file M/D/1 (temps de service déterministes).
        Équivalent à des temps de service Erlang-∞.
        
        Pour la distribution constante:
        - m₁ = temps_service
        - m₂ = temps_service²
        - CV² = 0
        """
        m1 = temps_service
        m2 = temps_service**2
        
        resultats = self.analyser_avec_moments(m1, m2)
        
        if "erreur" not in resultats:
            resultats["distribution"] = "Constante"
            resultats["interpretation"] = {
                "variabilite": "Nulle",
                "cv_carre_theorique": 0
            }
        
        return resultats

def afficher_resultats(resultats):
    """
    Fonction auxiliaire pour afficher les résultats d'analyse de file d'attente
    dans un format clair et pédagogique, utilisant des fractions.
    """
    if "erreur" in resultats:
        print("\nErreur:", resultats["erreur"])
        return
        
    print("\nRésultats de l'Analyse de la File d'Attente")
    print("=" * 50)
    
    # Métriques de stabilité
    print("\nMétriques de Stabilité:")
    print(f"Intensité du Trafic (ρ) = λE[X]: {format_fraction(resultats['intensite_trafic'])}")
    print(f"Utilisation du Serveur: {format_fraction(resultats['utilisation'] * 100)}%")
    
    # Métriques temporelles
    print("\nMétriques Temporelles:")
    print(f"E[X]  (Temps de Service Moyen): {format_fraction(resultats['E[X]'])}")
    print(f"E[Wq] (Temps d'Attente Moyen): {format_fraction(resultats['E[Wq]'])}")
    print(f"E[R]  (Temps de Réponse Moyen): {format_fraction(resultats['E[R]'])}")
    
    # Métriques de longueur de file
    print("\nMétriques de Longueur de File:")
    print(f"E[Qq] (Longueur Moyenne de la File): {format_fraction(resultats['E[Qq]'])}")
    print(f"E[Q]  (Taille Moyenne du Système): {format_fraction(resultats['E[Q]'])}")
    
    # Métriques de variabilité
    print("\nMétriques de Variabilité:")
    print(f"E[X²] (Second Moment du Service): {format_fraction(resultats['E[X²]'])}")
    print(f"CV²   (Coefficient de Variation au Carré): {format_fraction(resultats['CV²'])}")
    print(f"CV    (Coefficient de Variation): {format_fraction(resultats['CV'])}")
    
    # Informations spécifiques à la distribution
    if "distribution" in resultats:
        print(f"\nDétails de la Distribution ({resultats['distribution']}):")
        if "parametre_forme" in resultats:
            print(f"Paramètre de Forme (k): {format_fraction(resultats['parametre_forme'])}")
        if "parametres" in resultats:
            for param_name, param_value in resultats["parametres"].items():
                if isinstance(param_value, list):
                    formatted_values = [format_fraction(v) for v in param_value]
                    print(f"{param_name}: {formatted_values}")
                else:
                    print(f"{param_name}: {format_fraction(param_value)}")
        if "interpretation" in resultats:
            interp = resultats["interpretation"]
            print(f"\nCaractéristiques du Processus:")
            for key, value in interp.items():
                if isinstance(value, (int, float)):
                    print(f"- {key.capitalize()}: {format_fraction(value)}")
                else:
                    print(f"- {key.capitalize()}: {value}")

def tester_distributions(taux_arrivee=1.0, temps_service_moyen=0.5):
    """
    Fonction de test qui compare différentes distributions de service.
    """
    print(f"\nTest avec taux d'arrivée λ = {taux_arrivee} et temps de service moyen = {temps_service_moyen}")
    print("=" * 80)
    
    file = FileAttenteMG1(taux_arrivee)
    
    # Liste des tests à effectuer
    tests = [
        # Service constant (M/D/1)
        # ("Constant", lambda: file.analyser_service_constant(temps_service_moyen)),
        # 
        # # Service exponentiel (M/M/1)
        # ("Exponentiel", lambda: file.analyser_service_exponentiel(temps_service_moyen)),
        # 
        # Services Erlang-k
        ("Erlang-2", lambda: file.analyser_service_erlang(temps_service_moyen, 2)),
        # ("Erlang-4", lambda: file.analyser_service_erlang(temps_service_moyen, 4)),
        
        # # Services Hyper-exponentiels
        # ("Hyper-exp-2 (équiprobable)", 
        #  lambda: file.analyser_service_hyperexp(temps_service_moyen, 2)),
        # ("Hyper-exp-2 (p=[0.7,0.3])", 
        #  lambda: file.analyser_service_hyperexp(
        #      temps_service_moyen, 
        #      2, 
        #      p=np.array([0.7, 0.3]),
        #      mu=np.array([3/temps_service_moyen, 1/temps_service_moyen])
        #  )),
        # 
        # # Services Hypo-exponentiels
        # ("Hypo-exp-2 (auto)", 
        #  lambda: file.analyser_service_hypoexp(temps_service_moyen, 2)),
        # ("Hypo-exp-2 (taux fixés)", 
        #  lambda: file.analyser_service_hypoexp(
        #      temps_service_moyen,
        #      2,
        #      mu=np.array([3/temps_service_moyen, 2/temps_service_moyen])
        #  ))
    ]
    
    resultats_tests = []
    for nom, test in tests:
        print(f"\nTest de la distribution: {nom}")
        print("-" * 40)
        try:
            resultats = test()
            resultats["nom_test"] = nom
            resultats_tests.append(resultats)
            afficher_resultats(resultats)
        except Exception as e:
            print(f"Erreur lors du test: {str(e)}")
    
    return resultats_tests

def comparer_distributions(resultats_tests):
    """
    Crée des graphiques comparant les différentes distributions.
    """
    # Extraire les noms et métriques
    noms = [r["nom_test"] for r in resultats_tests]
    cv_carres = [r["CV²"] for r in resultats_tests]
    temps_attente = [r["E[Wq]"] for r in resultats_tests]
    temps_reponse = [r["E[R]"] for r in resultats_tests]
    
    # Création de la figure avec sous-graphiques
    
    # Graphique des CV²
    

def generer_rapport_latex(resultats_tests):
    """
    Génère un rapport LaTeX comparant les différentes distributions.
    """
    latex = []
    
    # En-tête du document
    latex.append(r"\documentclass{article}")
    latex.append(r"\usepackage[utf8]{inputenc}")
    latex.append(r"\usepackage{booktabs}")
    latex.append(r"\usepackage{float}")
    latex.append(r"\title{Analyse Comparative des Distributions de Service M/G/1}")
    latex.append(r"\begin{document}")
    latex.append(r"\maketitle")
    
    # Introduction
    latex.append(r"\section{Introduction}")
    latex.append(r"Ce rapport présente une analyse comparative de différentes distributions " + 
                r"de temps de service dans une file M/G/1.")
    
    # Tableau des résultats
    latex.append(r"\section{Résultats Comparatifs}")
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"Distribution & $\rho$ & CV² & E[Wq] & E[R] & E[Q] & E[Qq] \\")
    latex.append(r"\midrule")
    
    for r in resultats_tests:
        ligne = (f"{r['nom_test']} & "
                f"{format_fraction(r['intensite_trafic'])} & "
                f"{format_fraction(r['CV²'])} & "
                f"{format_fraction(r['E[Wq]'])} & "
                f"{format_fraction(r['E[R]'])} & "
                f"{format_fraction(r['E[Q]'])} & "
                f"{format_fraction(r['E[Qq]'])} \\\\")
        latex.append(ligne)    

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Comparaison des métriques de performance}")
    latex.append(r"\end{table}")
    
    # Analyse des résultats
    latex.append(r"\section{Analyse}")
    latex.append(r"\subsection{Variabilité}")
    latex.append("Les coefficients de variation (CV²) montrent que:")
    latex.append(r"\begin{itemize}")
    for r in resultats_tests:
        latex.append(fr"\item {r['nom_test']}: CV² = {r['CV²']:.3f}")
    latex.append(r"\end{itemize}")
    
    # Conclusions
    latex.append(r"\section{Conclusion}")
    latex.append(r"Cette analyse montre l'impact significatif de la distribution " +
                r"du temps de service sur les performances de la file d'attente.")
    
    latex.append(r"\end{document}")
    
    return "\n".join(latex)

if __name__ == "__main__":
    print("Test complet des différentes distributions de service")
    print("=" * 50)
    
    # Paramètres de test
    taux_arrivee = 1.0  # 1 client par unité de temps en moyenne
    temps_service_moyen = 1/6  # Temps de service moyen de 1/6 unité de temps 
    
    # Exécuter les tests
    resultats = tester_distributions(taux_arrivee, temps_service_moyen)
    
    # Générer les comparaisons graphiques
    comparer_distributions(resultats)
    
    # Générer le rapport LaTeX
    rapport_latex = generer_rapport_latex(resultats)
    
    # Sauvegarder le rapport LaTeX
    with open("rapport_file_mg1.tex", "w", encoding="utf-8") as f:
        f.write(rapport_latex)
    
    print("\nTests terminés. Les résultats ont été affichés et un rapport LaTeX a été généré.")
