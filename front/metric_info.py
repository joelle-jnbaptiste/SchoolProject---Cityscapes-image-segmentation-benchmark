metric_info = {
"train_loss": {
    "title": "Training Loss",
    "description": """ 
La *train loss* représente l’erreur du modèle pendant la phase d’entraînement.  
Elle mesure à quel point les prédictions du modèle s’éloignent des masques de vérité terrain sur le **jeu d’entraînement** uniquement.

- Une train loss basse indique une bonne capacité à apprendre les patterns du dataset.
- Une loss plus élevée ne signifie pas nécessairement un mauvais modèle : cela dépend de la fonction de perte utilisée.
- Dans la segmentation, **les fonctions de perte peuvent être très différentes entre les architectures**, rendant les valeurs non comparables directement.
""",
    "comparison": """
### Analyse du graphe — DeepLabV3+ vs Mask2Former

Sur ce graphique :
- **DeepLabV3+** apparaît en bleu et affiche une train loss extrêmement faible (proche de 0).  
- **Mask2Former**, en rouge, démarre autour de ~20 et descend lentement vers ~15.

Cette différence ne signifie pas que DeepLabV3+ est meilleur :  
- Les deux modèles utilisent des **fonctions de perte différentes**.  
- La loss de Mask2Former intègre une perte panoptique complexe (avec plusieurs composantes), ce qui génère des valeurs plus élevées.

### Conclusion
- La **forme de la courbe** est plus informative que les valeurs absolues.  
- **Mask2Former** montre une décroissance progressive et stable → apprentissage cohérent.  
- **DeepLabV3+** atteint très vite une loss basse → fonction de perte plus simple.

**Les train_loss ne doivent pas être comparées en valeur absolue entre ces deux modèles.**  
C’est le comportement de la courbe qui compte.
""",
"alt_text": """DeepLabV3+ conserve une perte d’entraînement quasi nulle tout au long des 10 époques,
restant stable autour de 0. Mask2Former commence avec une perte élevée autour de 20, 
puis diminue régulièrement jusqu’à environ 14 à la fin de l’entraînement. 
La courbe Mask2Former descend progressivement alors que DeepLab reste extrêmement bas 
et stable. La différence entre les deux modèles est constante et très marquée, 
indiquant que DeepLab converge beaucoup plus rapidement sur ce jeu d'entraînement.
"""
},
    "val_loss": {
    "title": "Validation Loss",
    "description": """
La *validation loss* mesure l’erreur du modèle sur un **jeu de données non vu pendant l’entraînement**.  
Elle permet d’évaluer :
- la capacité du modèle à **généraliser**,
- la présence éventuelle de **surapprentissage (overfitting)**,
- la stabilité du modèle sur des données réelles.

Cependant, la valeur brute de la val_loss dépend **fortement de la fonction de perte** utilisée.  
Dans la segmentation :
- DeepLabV3+ utilise une perte relativement simple (ex : cross-entropy),
- Mask2Former utilise des **pertes composites** (matching, panoptic loss…), ce qui produit des valeurs beaucoup plus élevées.
""",
    "comparison": """
### Analyse du graphe — DeepLabV3+ vs Mask2Former

Sur ce graphique :
- **DeepLabV3+ (bleu)** a une validation loss extrêmement basse (proche de 0).
- **Mask2Former (rouge)** oscille autour de **19–20**, avec de très faibles variations au fil des epochs.

Ces valeurs **ne sont pas comparables directement**, car les fonctions de perte ne mesurent pas les mêmes quantités et n'ont pas la même échelle.

### Ce qu'il faut observer
- La **courbe Mask2Former** est *stable*, sans dérive ou explosion → bonne régularité.
- La val_loss de Mask2Former ne diminue pas beaucoup, ce qui est normal avec sa loss complexe.
- La **forme** (stabilité, absence d’overfitting) importe plus que la valeur.

### Conclusion
- **Impossible de comparer les deux modèles en valeur absolue sur la validation loss.**
- **Mask2Former** montre une convergence stable mais sur une échelle de perte totalement différente.
- **DeepLabV3+** affiche une val_loss très faible en raison d’une fonction de perte plus simple.

""",
"alt_text": """DeepLabV3+ maintient une perte de validation très basse et stable autour de 0.1. 
Mask2Former conserve une perte élevée, proche de 20, avec très peu de variations 
entre les époques. La différence entre les deux modèles reste très importante et 
presque constante. DeepLab présente une bien meilleure généralisation selon cette 
métrique, tandis que Mask2Former ne montre pas d’amélioration significative sur le 
jeu de validation.
"""

},


"miou": {
    "title": "Mean Intersection over Union (mIoU)",
    "description": """
Le *mIoU* est la **métrique standard** en segmentation d’images.  
Elle mesure le recouvrement entre la prédiction et le masque de vérité terrain.

### Comment lire le mIoU ?
- 1.0 → segmentation parfaite  
- 0.0 → aucune correspondance  
- >0.75 → excellent  
- >0.65 → très bon  
- <0.50 → insuffisant  

Le mIoU prend en compte toutes les classes et pénalise fortement les erreurs sur les frontières ou petites classes.  
C’est la **métrique la plus fiable** pour comparer deux modèles de segmentation.
""",
    "comparison": """
### Analyse du graphe — DeepLabV3+ vs Mask2Former

Sur le graphique :
- **Mask2Former (rouge)** démarre autour de **0.76** et reste très stable tout au long de l’entraînement.
- Il montre une légère oscillation autour de cette valeur, pour finir proche de **0.78**.
- **DeepLabV3+ (bleu)** progresse lentement de **0.60** jusqu’à environ **0.66**.

### Interprétation
- Mask2Former est **nettement supérieur** en mIoU, avec un avantage constant de **+10 à +12 points** sur tout l’entraînement.
- DeepLabV3+ progresse régulièrement mais reste loin derrière.
- La stabilité du mIoU de Mask2Former montre une segmentation **robuste** et **cohérente**, même à faible nombre d’epochs.

### Conclusion
- **Mask2Former est clairement le meilleur modèle** d’après le mIoU.  
- Il segmente mieux, sur toutes les classes, avec une meilleure constance.  
- DeepLabV3+ reste performant mais moins compétitif face à une architecture transformer moderne.


""",
"alt_text": """DeepLabV3+ commence avec une mIoU d’environ 0.59 et progresse lentement jusqu’à 
environ 0.66 sur 10 époques. Mask2Former démarre nettement plus haut, autour de 0.76, 
puis oscille légèrement avant de monter autour de 0.78 sur la dernière époque. 
Mask2Former surpasse DeepLab sur toute la durée de l’entraînement, avec une marge 
constante de 0.10 à 0.12 en mIoU.
"""

},

"pixel_acc": {
    "title": "Pixel Accuracy",
    "description": """
La *pixel accuracy* mesure le pourcentage de pixels correctement classés.

### Comment interpréter cette métrique ?
- C’est une métrique simple : **nombre de pixels corrects / total des pixels**.
- Elle est intuitive mais possède une limite importante :
  → si une classe est beaucoup plus présente (ex : "route"), un modèle peut obtenir une bonne accuracy **en ignorant les classes rares**.
  
### Points importants :
- pixel_acc est utile pour un aperçu global,
- mais elle ne remplace pas le **mIoU**, bien plus fiable pour juger la qualité d’une segmentation.
""",
    "comparison": """
### Analyse du graphe — DeepLabV3+ vs Mask2Former

Sur le graphique :
- **Mask2Former (rouge)** reste constamment entre **0.945 et 0.955**, une zone très haute.
- La courbe est stable, légèrement ascendante en fin d’entraînement.
- **DeepLabV3+ (bleu)** varie entre **0.91 et 0.94**, avec des fluctuations plus marquées.

### Interprétation
- Mask2Former atteint **2 à 3 points d’accuracy en plus** par rapport à DeepLabV3+ sur l’ensemble des epochs.
- Sa stabilité montre une meilleure capacité à classer correctement l’ensemble des pixels.
- Les variations de DeepLabV3+ indiquent une sensibilité plus forte aux mini-batches ou aux classes difficiles.

### Conclusion
 - **Mask2Former obtient une pixel accuracy systématiquement meilleure**.  
 - Cela confirme sa robustesse et son avantage sur la segmentation globale.  
 - DeepLabV3+ reste performant mais affiche plus de volatilité.


""",
"alt_text": """DeepLabV3+ progresse de 0.91 à environ 0.94 en précision pixel, avec quelques 
fluctuations. Mask2Former reste constamment au-dessus, autour de 0.95 à 0.955, avec 
une courbe stable et supérieure à celle de DeepLab pendant toute la durée. 
La différence entre les deux modèles reste faible mais constante, Mask2Former 
conservant la meilleure précision.
"""

},
"imgs_per_sec": {
    "title": "Images par seconde (Throughput)",
    "description": """
Cette métrique mesure la **vitesse d’inférence/entraînement** du modèle, c’est-à-dire :
combien d’images le modèle peut traiter chaque seconde.

### Pourquoi cette métrique est importante ?
- Elle indique la **rapidité** du modèle.
- Elle est utile pour estimer le **coût compute** (GPU/CPU).
- Elle a un impact direct sur :
  - le temps d'entraînement total,
  - le temps de prédiction dans une application réelle,
  - la scalabilité du modèle.

### Points clés à retenir :
- Un throughput plus élevé = un modèle plus rapide.
- Ce n’est **pas une métrique de performance**, mais une métrique d’**efficacité**.
""",
    "comparison": """
### Analyse du graphe — DeepLabV3+ vs Mask2Former

Sur ton graphique :
- **Mask2Former (rouge)** se situe en permanence entre **6.3 et 6.5 images/sec**.
- **DeepLabV3+ (bleu)** oscille plutôt entre **5.6 et 5.9 images/sec**.
- Les deux courbes sont stables au fil des epochs, mais Mask2Former garde un avantage constant d’environ **+0.5 à +0.7 images/sec**.

### Interprétation
- Mask2Former est **plus rapide** que DeepLabV3+, ce qui peut surprendre puisqu’il est plus complexe.
- Cela montre une très bonne **optimisation interne des architectures transformer modernes**.
- Cette avance en vitesse est significative :  
  → ~10% à ~12% de throughput en plus selon les epochs.

### Conclusion
- **Mask2Former est non seulement plus précis, mais aussi plus rapide.**  
- Il atteint un throughput supérieur tout au long de l’entraînement.  
- DeepLabV3+ reste performant mais plus lent, ce qui augmente légèrement le coût compute.

Globalement : Mask2Former offre un **meilleur rapport vitesse / précision** que DeepLabV3+.
""",
"alt_text": """DeepLabV3+ traite environ 5.6 à 5.9 images par seconde, avec de légères variations.
Mask2Former est systématiquement plus rapide, oscillant entre 6.3 et 6.5 images par 
seconde. Mask2Former est donc environ 10 à 15 % plus rapide que DeepLab tout au long 
de l’entraînement.
"""

},

"train_time_sec": {
    "title": "Temps d'entraînement par epoch (train_time_sec)",
    "description": """

Cette métrique indique **combien de secondes sont nécessaires pour entraîner un modèle sur un epoch complet**.

Elle donne une mesure directe :
- de la **vitesse globale du modèle**,  
- de son **coût computationnel**,  
- de son impact sur le temps total d'entraînement.

### À quoi sert cette métrique ?
- Comparer l’efficacité des architectures.
- Estimer le temps total nécessaire pour un entraînement complet.
- Évaluer la faisabilité du modèle dans un contexte réel (inférence ou réentraînement régulier).

### Attention :
- Cette métrique dépend du modèle mais aussi :  
  → du matériel (GPU/CPU),  
  → du batch size,  
  → de la complexité interne du modèle.

Elle complète bien les métriques de throughput (`imgs_per_sec`).
""",
    "comparison": """
### Analyse du graphe — DeepLabV3+ vs Mask2Former

Sur le graphique :
- **DeepLabV3+ (bleu)** tourne autour de **505–530 sec/epoch**.  
  Sa courbe montre de petites oscillations mais reste globalement stable.
- **Mask2Former (rouge)** est plus rapide, entre **460–475 sec/epoch**, avec une stabilité remarquable.

### Interprétation
- Mask2Former nécessite environ **35 à 55 secondes de moins** par epoch que DeepLabV3+.
- Cela représente un **gain de temps de 8% à 12%** par epoch.
- Ce résultat peut sembler surprenant : Mask2Former est plus complexe, mais :
  - ses opérations vectorisées et transformer-friendly s'exécutent très efficacement sur GPU,
  - certaines étapes de DeepLabV3+ (dense convolutions) sont plus coûteuses.

### Conclusion
- **Mask2Former s'entraîne plus vite** que DeepLabV3+.  
- Il est non seulement plus précis, mais aussi plus efficace en temps de calcul.  
- DeepLabV3+ reste compétitif mais demande plus de temps par epoch.

Couplé à `imgs_per_sec`, cela montre que **Mask2Former surpasse DeepLabV3+ en vitesse ET en qualité**.
""",
"alt_text": """DeepLabV3+ affiche un temps d’entraînement par époque autour de 510 à 530 secondes. 
Mask2Former est plus rapide, avec environ 460 à 475 secondes par époque. 
Mask2Former reste toujours plus performant en temps d'entraînement, avec une 
différence moyenne d’environ 50 secondes par époque.
"""

},

}