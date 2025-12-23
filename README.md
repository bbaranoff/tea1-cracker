# TEA1 Key Brute-forcer (OpenCL Accelerated)

Ce projet est un outil de recherche de clé pour l'algorithme de chiffrement **TEA1**. Il utilise la puissance de calcul parallèle des processeurs graphiques (**GPU**) via **OpenCL** pour tester l'intégralité de l'espace des clés (32 bits) en un temps record.

## 📖 Principe de fonctionnement

Le script repose sur une attaque par **force brute à texte clair connu (Known Plaintext Attack)**. Si vous disposez d'un fragment du flux chiffré et que vous connaissez (ou devinez) le contenu original, vous pouvez isoler le **Keystream**.

### 1. Inversion du Keystream

TEA1 est un chiffrement de flux. Le processus est le suivant :



Le script prend en entrée 64 bits (16 caractères hexadécimaux) de ce keystream pour valider si une clé candidate est la bonne.

### 2. Algorithme de recherche

* **Initialisation de l'IV** : Le script reconstruit l'Instruction Vector (IV) à partir des paramètres de trame (Timeslot, Frame Number, etc.) via la fonction `build_iv`.
* **Parallélisation GPU** : L'espace de recherche de  clés est divisé en paquets (batches). Le kernel OpenCL teste simultanément des milliers de clés.
* **Validation 64-bit** : Contrairement aux versions simplifiées qui testent 32 bits, ce script vérifie 64 bits du keystream pour éliminer les "fausses alertes" (collisions) et garantir que la clé trouvée est l'unique clé correcte.

---

## 🚀 Utilisation

### Prérequis

* Un GPU compatible OpenCL.
* Python 3.x avec les bibliothèques : `pyopencl`, `numpy`.

### Syntaxe

Le script requiert les paramètres réseau de la trame interceptée pour synchroniser l'état interne de l'algorithme.

```bash
python crack_tea1.py <TN> <HN> <MN> <FN> <SN> <Direction> <Keystream_Hex>

```

**Arguments :**

* `TN`, `HN`, `MN`, `FN` : Numéros de trames et slots (Time/Hyper/Macro/Frame numbers).
* `Direction` : 0 ou 1 (Uplink/Downlink).
* `Keystream_Hex` : Les 16 premiers caractères hexadécimaux du keystream extrait.

**Exemple :**

```bash
python tea1_opencl_crack.py 1 110 30 06 1 0 0BE7FE9AE1EA459F866919C9E2EA1E11A77A4493D658A4191EDD987F37DE12B1DA3F7BBD62607E8CE787C2FE544B2FAAEAED38255BEB
```

---

## ⚡ Performance et Impact

### Impact Technique

* **Vitesse** : Sur un GPU de milieu de gamme, l'intégralité de l'espace de clé 32 bits peut être parcourue en quelques minutes (voire secondes), contre plusieurs heures sur un CPU classique.
* **Sécurité** : Cet outil démontre la faiblesse critique de TEA1. Avec une clé de seulement 32 bits d'entropie effective, le chiffrement ne résiste pas à une analyse computationnelle moderne.

### Limites

* **Accès au Keystream** : L'utilisateur doit être capable d'identifier au moins 8 octets de données connues (comme des en-têtes LLC ou IP) pour extraire le keystream.
* **Matériel** : La performance dépend directement du nombre d'unités de calcul (Compute Units) du GPU utilisé.

---

## ⚠️ Avertissement Légal

Cet outil est fourni à des fins **éducatives et de recherche en cybersécurité** uniquement. L'interception et le décodage de communications privées sans autorisation sont illégaux dans la plupart des juridictions. L'utilisateur est seul responsable de l'usage qu'il fait de ce logiciel.
