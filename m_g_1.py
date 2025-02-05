import numpy as np

class FileAttenteMG1:
    """
    Analyse d'une file d'attente M/G/1 utilisant la formule de Pollaczek-Khinchin.
    
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
                "phases": f"Le processus de service consiste en {k} phases exponentielles",
                "variabilite": "Faible" if k > 5 else "Moyenne" if k > 1 else "Élevée",
                "cv_carre_theorique": 1/k
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
        """
        return self.analyser_avec_moments(temps_service, 0)

def afficher_resultats(resultats):
    """
    Fonction auxiliaire pour afficher les résultats d'analyse de file d'attente
    dans un format clair et pédagogique.
    """
    if "erreur" in resultats:
        print("\nErreur:", resultats["erreur"])
        return
        
    print("\nRésultats de l'Analyse de la File d'Attente")
    print("=" * 50)
    
    # Métriques de stabilité
    print("\nMétriques de Stabilité:")
    print(f"Intensité du Trafic (ρ) = λE[X]: {resultats['intensite_trafic']:.8f}")
    print(f"Utilisation du Serveur: {resultats['utilisation']:.2%}")
    
    # Métriques temporelles
    print("\nMétriques Temporelles:")
    print(f"E[X]  (Temps de Service Moyen): {resultats['E[X]']:.8f}")
    print(f"E[Wq] (Temps d'Attente Moyen): {resultats['E[Wq]']:.8f}")
    print(f"E[R]  (Temps de Réponse Moyen): {resultats['E[R]']:.8f}")
    
    # Métriques de longueur de file
    print("\nMétriques de Longueur de File:")
    print(f"E[Qq] (Longueur Moyenne de la File): {resultats['E[Qq]']:.8f}")
    print(f"E[Q]  (Taille Moyenne du Système): {resultats['E[Q]']:.8f}")
    
    # Métriques de variabilité
    print("\nMétriques de Variabilité:")
    print(f"E[X²] (Second Moment du Service): {resultats['E[X²]']:.8f}")
    print(f"CV²   (Coefficient de Variation au Carré): {resultats['CV²']:.8f}")
    print(f"CV    (Coefficient de Variation): {resultats['CV']:.8f}")
    
    # Informations spécifiques à la distribution Erlang
    if "distribution" in resultats and resultats["distribution"].startswith("Erlang"):
        print(f"\nDétails de la Distribution Erlang:")
        print(f"Paramètre de Forme (k): {resultats['parametre_forme']}")
        interp = resultats["interpretation"]
        print(f"Caractéristiques du Processus:")
        print(f"- {interp['phases']}")
        print(f"- Niveau de Variabilité: {interp['variabilite']}")
        print(f"- Taux de Phase (μ): {resultats['taux_phase']:.8f}")

if __name__ == "__main__":
    # Exemple d'utilisation avec différents temps de service Erlang-k
    taux_arrivee = 1.0  # Moyenne d'1 client par unité de temps
    temps_service_moyen = 1/6  # Temps de service moyen de 0.5 unités de temps
    file = FileAttenteMG1(taux_arrivee)
    
    # Comparaison de différents services Erlang-k
    valeurs_k = [2]
    
    for k in valeurs_k:
        print(f"\nFile M/G/1 avec Temps de Service Erlang-{k}")
        print("=" * 50)
        resultats = file.analyser_service_erlang(temps_service_moyen, k)
        afficher_resultats(resultats)
