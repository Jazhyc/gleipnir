import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rows = {
    json.loads(l)["id"]: json.loads(l)
    for l in open("data/adversarial_lab_liars/main_student.jsonl")
}
oof = [json.loads(l) for l in open("results/adversarial_lab_liars_probe/oof_4b.jsonl")]
tr = [
    o
    for o in oof
    if o["role"] == "recruiter" and o["condition"] in ("liar", "honest_recruiter")
]
text = lambda i: rows[i]["prompt"].split("<agent_trajectory>")[-1]
seeds = sorted({o["seed"] for o in tr})
score = {}
for s in seeds:
    fit = [o for o in tr if o["seed"] != s]
    test = [o for o in oof if o["seed"] == s and o["role"] == "recruiter"]
    v = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    X = v.fit_transform([text(o["id"]) for o in fit])
    y = [o["condition"] == "liar" for o in fit]
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    for o, p in zip(
        test, clf.decision_function(v.transform([text(o["id"]) for o in test]))
    ):
        score[o["id"]] = p
prim = [
    o
    for o in oof
    if o["analysis_recruiter"]
    and o["cell"] in ("main", "haiku", "sol")
    and o["condition"] in ("liar", "honest_recruiter")
]
y = [o["condition"] == "liar" for o in prim]
print("primary rows: %d liars, %d honest" % (sum(y), len(y) - sum(y)))
print(
    "word-count (TF-IDF) baseline AUROC: %.3f"
    % roc_auc_score(y, [score[o["id"]] for o in prim])
)
print("4B probe AUROC: %.3f" % roc_auc_score(y, [o["probe"] for o in prim]))
