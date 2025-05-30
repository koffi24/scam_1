# **Rapport du Projet SCAM_1**

---

## **1. Objectif du Projet**
Le projet `SCAM_1` a pour objectif de développer une application interactive permettant de simuler une victime d'arnaque dans le but de faire perdre du temps à un arnaqueur. L'application repose sur une interface web où deux rôles sont définis :

- **User1 (Arnaqueur)** : Génère des réponses automatiques basées sur un modèle LLM (Large Language Model).
- **User2 (Victime)** : Interagit avec l'arnaqueur en posant des questions et en simulant un intérêt pour l'arnaque.

L'application utilise Flask et Flask-SocketIO pour gérer les interactions en temps réel, ainsi que des services de reconnaissance vocale et de synthèse vocale pour enrichir l'expérience utilisateur.

---

## **2. Fonctionnalités Implémentées**

### **2.1. Interface Web**
- Une interface utilisateur simple et intuitive a été développée avec HTML, CSS et JavaScript.
- Les utilisateurs peuvent choisir leur rôle (arnaqueur ou victime) et échanger des messages en temps réel.
- Les messages sont affichés avec des styles distincts pour chaque rôle.

### **2.2. Backend Flask**
- Un serveur Flask gère les routes et les événements SocketIO pour la communication en temps réel.
- Les messages envoyés par la victime (`user2`) déclenchent une réponse automatique de l'arnaqueur (`user1`) en utilisant un modèle LLM.

### **2.3. Modèle LLM**
- Le modèle LLM (via `InferenceClient`) génère des réponses réalistes pour l'arnaqueur en suivant un contexte, des instructions et des contraintes spécifiques.

### **2.4. Reconnaissance et Synthèse Vocale**
- La reconnaissance vocale (via Google Cloud Speech-to-Text) permet de convertir les messages audio en texte.
- La synthèse vocale (via Google Cloud Text-to-Speech) génère des réponses audio pour rendre l'interaction plus immersive.

---

## **3. Structure du Projet**
Le projet est organisé comme suit :


---

## **4. Technologies Utilisées**

### **Backend :**
- Flask
- Flask-SocketIO
- Google Cloud Speech-to-Text
- Google Cloud Text-to-Speech
- Hugging Face Inference Client

### **Frontend :**
- HTML, CSS, JavaScript

### **Autres :**
- SoundDevice et SoundFile pour la gestion des fichiers audio.

---

## **5. Étapes Réalisées**

### **5.1. Configuration de l'Environnement**
- Création d'un environnement virtuel Python (`.venv`).
- Installation des dépendances nécessaires (Flask, Flask-SocketIO, Google Cloud SDK, etc.).

### **5.2. Développement Backend**
- Mise en place du serveur Flask pour gérer les routes et les événements SocketIO.
- Intégration du modèle LLM pour générer des réponses automatiques.
- Ajout des fonctionnalités de reconnaissance et de synthèse vocale.

### **5.3. Développement Frontend**
- Création de l'interface utilisateur avec `index.html`.
- Ajout de styles CSS pour différencier les messages des deux rôles.
- Intégration de JavaScript pour gérer les événements SocketIO.

### **5.4. Exclusion des Fichiers Sensibles**
- Création d'un fichier `.gitignore` pour exclure le dossier `.venv`, les clés API (`key.json`) et les fichiers audio générés.

### **5.5. Tests et Débogage**
- Vérification du bon fonctionnement de l'interface web.
- Tests des interactions en temps réel entre les deux rôles.
- Validation des fonctionnalités de reconnaissance et de synthèse vocale.

### **5.6. Documentation**
- Rédaction d'un fichier `README.md` pour documenter le projet.

---

## **6. Résultats**
- L'application est fonctionnelle et permet une interaction fluide entre la victime et l'arnaqueur.
- Les réponses générées par le modèle LLM respectent le contexte et les instructions définies.
- L'interface web est intuitive et différencie clairement les messages des deux rôles.

---

## **7. Prochaines Étapes**

### **Améliorations Fonctionnelles :**
- Ajouter un système de journalisation pour enregistrer les conversations.
- Permettre le téléchargement des conversations sous forme de fichier texte.

### **Optimisation :**
- Réduire la latence des réponses générées par le modèle LLM.
- Optimiser la gestion des fichiers audio pour réduire l'utilisation des ressources.

### **Sécurité :**
- Chiffrer les communications entre le client et le serveur.
- Ajouter une gestion des erreurs pour les services vocaux et le modèle LLM.

---

## **8. Conclusion**
Le projet `SCAM_1` est une application innovante qui combine des technologies modernes pour simuler des interactions réalistes entre une victime et un arnaqueur. Grâce à l'intégration de modèles LLM et de services vocaux, l'application offre une expérience immersive tout en respectant les contraintes de sécurité et de confidentialité.

---

## **9. Commandes Clés**

### **Pour exécuter le projet :**

python [app.py](http://_vscodecontentref_/4)

Pour installer les dépendances :
pip install -r [requirements.txt](http://_vscodecontentref_/5)


Pour configurer les clés API :
Définir les variables d'environnement pour les clés API (NEBIUS_API_KEY, GOOGLE_APPLICATION_CREDENTIALS).