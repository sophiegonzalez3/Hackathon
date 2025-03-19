import os
import simulate
import numpy as np

#Permet d'entraîner le modèle sur l'ensemble des scénarios générés dans le dossier eval_configs
def train_scenarios():
    scenario_paths = sorted([
        os.path.join("eval_configs", f)
        for f in os.listdir("eval_configs") if f.startswith("config_")
    ])

    for i, config_path in enumerate(scenario_paths, 1):
        print(f"\nEntraînement sur scénario {i}: {config_path}")
        
        # Entraîner avec le chemin du fichier de configuration
        trained_agent, all_rewards = simulate.train(config_path)
        print(f"Moyennes des récompenses obtenues pour le scénario {i} : {np.mean(all_rewards)}")
