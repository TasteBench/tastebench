##ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.

**[See Section 3.18 for defined terms](rules#18.-terms)**

*The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website.*

**You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.**

<h3>1. COMPETITION-SPECIFIC TERMS</h3>
<h4>1. COMPETITION TITLE</h4> Molecular Taste Classification
<h4>2. COMPETITION SPONSOR</h4> TasteBench Organizers
<h4>3. COMPETITION SPONSOR ADDRESS</h4> Not applicable — non-monetary academic benchmark hosted by the TasteBench Organizers.
<h4>4. COMPETITION WEBSITE</h4> https://www.kaggle.com/competitions/[INSERT: competition slug after launch]
<h4>5. TOTAL PRIZES AVAILABLE</h4> None. This is a non-monetary, community-driven academic benchmark; no cash prizes are awarded.
<h4>6. WINNER LICENSE TYPE</h4> Not applicable — no prizes are awarded. Top-ranked entries are strongly encouraged (but not required) to publish their training code under an OSI-approved open-source license.
<h4>7. DATA ACCESS AND USE</h4> Competition Use and Commercial. The Competition Data is redistributed from FartDB (Zimmermann et al. 2024) under the MIT License and may be used for any purpose, commercial or non-commercial, including participating in the Competition, on Kaggle.com forums, and for academic research and education. The Competition Data is also subject to the upstream FartDB license terms: https://github.com/fart-lab/fart (commit `bde90e6562ce5d248e76af791fab29ffc9ae901b`).

###2. COMPETITION-SPECIFIC RULES
In addition to the provisions of the General Competition Rules below, you understand and agree to these Competition-Specific Rules required by the Competition Sponsor:

####1. TEAM LIMITS
a. The maximum Team size is five (5).
b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running.

####2. SUBMISSION LIMITS
a. You may submit a maximum of five (5) Submissions per day.
b. Because the leaderboard is 100% Public (no private-leaderboard reshuffle), the final score is the best Submission across all submissions made by your Team during the Competition. There is no separate Final Submission selection step.

####3. COMPETITION TIMELINE
a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition's Overview > Timeline page.

####4. COMPETITION DATA

a. Data Access and Use. You may access and use the Competition Data for any purpose, whether commercial or non-commercial, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules. The Competition Data is also subject to the upstream FartDB MIT license: https://github.com/fart-lab/fart at commit `bde90e6562ce5d248e76af791fab29ffc9ae901b`.

b. Data Security. You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree to notify Kaggle immediately upon learning of any possible unauthorized access to the Competition Data and agree to work with Kaggle to rectify any such access.

c. Training Data Use. Participants may train on `train.csv` alone, or on `train.csv` and `val.csv` combined. The published FART benchmark (Zimmermann et al. 2024) and the TasteBench GNN reference were both computed training on `train.csv` only, with `val.csv` used for early stopping and hyperparameter selection. Submissions trained on the combined set are eligible but should disclose this in any associated write-up so readers can interpret leaderboard comparisons correctly.

####5. WINNER LICENSE
Not applicable. No prizes are awarded in this Competition; therefore no Winner License is required as a condition of participation. Top-ranked entries are strongly encouraged to publish their training code under an OSI-approved open-source license, but doing so is not a condition of participation or recognition.

####6. EXTERNAL DATA AND TOOLS

a. Use of External Data is permitted under the following conditions:

1. **Pretraining on unlabeled molecular corpora** (e.g., ChEMBL, PubChem, ZINC) is allowed and encouraged.

2. **Public pretrained models** (e.g., ChemBERTa, MolFormer, MoLFormer-XL, Uni-Mol) are allowed.

3. **Additional labeled taste/flavor data** beyond the provided FartDB splits — including but not limited to BitterDB, ChemTastesDB updates, and post-FART labeled molecules from published papers — is **allowed**, provided that all external labeled sources are disclosed in any associated write-up or code release submitted by the Participant.

b. Reasonableness Standard. Per Section 2.6.b of the General Rules, External Data and Tools must be reasonably accessible to all Participants. Use of restricted, proprietary, or paywalled labeled taste data is permitted only if equivalent access is reasonably available to other Participants.

c. Automated Machine Learning Tools (AMLT). Individual Participants and Teams may use automated machine learning tools to create a Submission, provided they have an appropriate license to comply with the Competition Rules.

####7. ELIGIBILITY

a. Participation is open to all Kaggle users in good standing, subject to Kaggle's standard account and conduct policies. As no monetary prizes are awarded, no special eligibility constraints beyond Kaggle's standard rules apply.

####8. HONOR SYSTEM ON THE PUBLIC BENCHMARK

a. The test-set labels for this Competition are openly available in the upstream FartDB GitHub repository (https://github.com/fart-lab/fart) at the commit this Competition mirrors. The Competition Sponsor acknowledges this and intends this Competition as a transparent benchmark, not a hidden-label challenge.

b. Submissions found to use direct test-label lookups — including but not limited to joining the test-set canonicalized SMILES against the upstream FartDB to recover ground-truth labels — will be removed from public rankings.

c. By submitting, Participants represent in good faith that their predictions were generated by a model trained on Competition Data and any disclosed External Data, and not derived from looking up the test labels in the upstream FartDB.

d. Submissions that produce a leaderboard score without a corresponding reproducible training pipeline are **not recognized** as benchmark contributions and may be removed from public rankings at the Competition Sponsor's discretion.

e. The Competition Sponsor reserves the right to request runnable training code from any Submission whose leaderboard score is inconsistent with its disclosed methodology. Submissions that cannot produce a reproducible pipeline upon request may be excluded from public rankings.

####9. CITATION REQUIREMENT

a. Any publication, preprint, presentation, or technical report based on this Competition's data must cite the upstream FartDB paper:

```bibtex
@unpublished{Zimmermann2024chemical,
    doi       = {10.26434/chemrxiv-2024-d6n15-v2},
    publisher = {American Chemical Society (ACS)},
    title     = {A Chemical Language Model for Molecular Taste Prediction},
    url       = {http://dx.doi.org/10.26434/chemrxiv-2024-d6n15-v2},
    author    = {Zimmermann, Yoel and Sieben, Leif and Seng, Henrik and Pestlin, Philipp and G{\"o}rlich, Franz},
    date      = {2024-12-11},
}
```

####10. GOVERNING LAW

a. All claims arising out of or relating to these Rules will be governed by California law, excluding its conflict of laws rules, and will be litigated exclusively in the Federal or State courts of Santa Clara County, California, USA. The parties consent to personal jurisdiction in those courts. If any provision of these Rules is held to be invalid or unenforceable, all remaining provisions of the Rules will remain in full force and effect.
