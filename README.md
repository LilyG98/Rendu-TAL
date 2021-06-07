<h4>Lily Grumbach
<br>M1 Humanités numériques - Université PSL</h4>
    <h1><center>Rendu TAIS-TAL </center></h1>
    <h2><center>Partie TAL</center></h2>
    <h3><center>2/3 : Reconnaissance d'entités nommées sur Doccano</center></h3>
    <h4><center>PLAN</center></h4>
    
    1) Exploration des données

    <b>2) NER avec Doccano</b>
    
        a) initialisation des données
        b) Exploration de la NER préalable de Spacy
        c) Mise en place de la stratégie pour l'annotation sur Doccano
        d) Entraînement par 5-fold cross validation
    
    3) Application du NER au df
    
    <b>Enjeux de la partie 2:</b>
    Les deux revues scientifiques étudiées reposent sur trois éléments consitutifs:
    - <b>Thématiques</b> : 
        - Hygiène 
        - Pathologies dites \exotiques\ ou \coloniales\
        - Liens avec les sciences humaines et sociales
        - Relations avec les découvertes récentes
    - <b>Terrains spécifiques</b>  :
        - Colonies et protectorats français
        - Batiments navals de la marine française
        - Autres territoires hors métropole et possessions françaises
        - Dans certains cas des laboratoires
    - <b>Corps de métier</b> Population particulière de rédacteurs que sont les membres du corps de santé des colonies et du corps de médecine navale(fait l'objet d'une analyse de réseau non prise en considération dans ce devoir),
   
    
    <b>Objectifs de la partie 2:</b>
    - Entraîner une NER sur notre BDD
    
    <b> L'objectif à long terme </b> de cet entraînement NER est de :
    - Mettre en place une NER qui pourrait être appliquée au full text de chacune des revues une fois les OCR nettoyées.
    - Appliquer cette NER à la revue des Archives de médecine navale et coloniale (1892-1898)
    - Alimenter une future base de données relationnelle
    
    
    <b>Limites de mon travail:</b> Il me sera nécessaire de mesurer si elle fonctionne bien à la fois sur le reste de la période étudiée ainsi que sur le Bulletin de la Société de Pathologie exotique

</text>
