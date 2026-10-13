# Rapport Ragas — `ragas_evaluation.ipynb`

**Source** : exécution du notebook (cellule d’évaluation, `execution_count` 18) sur `rag_predictions.json`.  
**Juge** : `gpt-4o-mini` (température 0) + embeddings `text-embedding-3-small`.  
**Échantillon** : **7** questions (réponses et contextes FAISS pré-calculés dans le JSON).

---

## Métriques 

- **faithfulness** : mesure si la réponse invente des faits absents des contextes récupérés.
- **answer_relevancy** : mesure si la réponse traite bien la question posée.
- **context_precision** : mesure si les segments FAISS sont pertinents pour produire la réponse de référence.
- **answer_correctness** : proximité sémantique entre réponse générée et réponse annotée.

---

# Run 1 - 2026-10-12

## Scores moyens

| Métrique | Moyenne | Lecture rapide |
|----------|---------|----------------|
| **faithfulness** | **0,85** | Les réponses s’appuient en grande partie sur les contextes fournis (peu d’inventions flagrantes). |
| **answer_relevancy** | **0,56** | Pertinence vis-à-vis de la question : hétérogène ; deux refus « hors sujet » tirent la moyenne vers le bas. |
| **context_precision** | **0,76** | Les blocs récupérés sont souvent utiles pour la référence, mais pas toujours pour la question posée. |
| **context_recall** | **0,43** | Une part importante des faits attendus (réponse annotée) n’apparaît pas dans les 5 segments FAISS. |

**Taux « acceptable »** (maison : *faithfulness* ≥ 0,7 **et** *answer_relevancy* ≥ 0,7) : **3 / 7 (42,9 %)** — Q05, Q06, Q07.

---

## Détail par question

| # | Question (extrait) | faith. | ans. rel. | ctx prec. | ctx recall | Verdict |
|---|-------------------|--------|-----------|-----------|------------|---------|
| Q01 | Atelier métiers du numérique | 0,92 | 0,70 | 1,00 | 0,50 | Bon match ; *answer_relevancy* juste sous 0,7 ; recall partiel vs référence. |
| Q02 | Rendez-vous agence YES (Borderouge) | 0,67 | **0,00** | 1,00 | **0,00** | **Échec retrieval** : refus honnête ; FAISS renvoie d’autres job datings. |
| Q03 | Job Dating Minimes / Liebherr | 0,67 | **0,00** | **0,00** | **0,00** | **Échec retrieval** : contextes proches de Q02 ; cible absente du top-5. |
| Q04 | Atelier création d’entreprise | 0,88 | 0,65 | 1,00 | **0,00** | Bon événement (10 questions créateurs) mais **autre atelier** que la réf. annotée → recall nul. |
| Q05 | Concert Fazil Say (Jacobins) | 0,80 | 0,94 | 0,33 | 0,50 | Réponse pertinente ; **contexte bruité** (précision basse). |
| Q06 | KIWI Indus/Logistique à Borderouge | 1,00 | 0,78 | 1,00 | 1,00 | **Référence** : retrieval + génération alignés. |
| Q07 | Récital Elisabeth Brauss (Jacobins) | 1,00 | 0,83 | 1,00 | 1,00 | **Succès** complet sur les quatre métriques. |

---

## Interprétation synthétique

### Ce qui fonctionne

- **Génération prudente** : *faithfulness* élevé en moyenne (0,85) — le prompt « n’inventez pas » est globalement respecté ; refus explicites quand l’événement n’est pas dans les contextes (Q02, Q03).
- **Requêtes factuelles bien indexées** : emploi KIWI (Q05) et récital Brauss (Q06) atteignent des scores proches du plafond sur les quatre métriques.
- **Atelier numérique (Q01)** : bon match sémantique malgré du bruit (« atelier numérique » bibliothèques) dans le top-5.

### Points faibles dominants

1. **Retrieval sur requêtes « entité + lieu + marque »** (YES, Liebherr, Minimes) : l’index par **description entière** et la similarité embedding ne discriminent pas assez les job datings ; le modèle reçoit des contextes **homogènes mais faux**, d’où refus corrects côté génération mais *answer_relevancy* = 0 (réponse non alignée avec ce que l’utilisateur attendait).
2. **Context recall modéré (0,43)** : même quand la réponse est bonne, les segments FAISS ne couvrent pas tout le contenu de la réponse annotée.
3. **Ambiguïté?** (Q04) : *context_precision* = 0,33 signale du bruit dans le retrieval ; requêtes « Jacobins / piano » pourraient attirer des événements voisins non pertinent.

--- 

## Piste à prioriser

Les descriptions indexées font **au plus 200 caractères**. De plus **seule `description_fr` est vectorisée** (ni le titre, ni le lieu, ni les dates, ni la description longue). Le contexte envoyé au LLM contient bien toute la ligne, mais la phase de **recherche** ne « voit » qu'une phrase courte, ce qui pourrait expliquer les difficultés de récupération des informations pertinentes dans la base de données.