import json,argparse

def read(p):
    with open(p,encoding="utf8") as f:
        return [json.loads(x) for x in f if x.strip()]

def gold(p):
    out=[]
    for r in read(p):
        for e in r["entities"]:
            out.append({
                "note_id":str(r["note_id"]),
                "start":int(e["start"]),
                "end":int(e["end"]),
                "text":e.get("text",""),
                "snomed_id":str(e.get("snomed_id","")),
                "cui":e.get("cui")
            })
    return out

def pred(p,fmt):
    out=[]
    for r in read(p):
        if "entities" in r:
            es=r["entities"]
            nid=str(r["note_id"])
        elif "prediction" in r:
            es=r["prediction"].get("entities",[])
            nid=str(r["note_id"])
        else:
            es=[r]
            nid=str(r["note_id"])

        for e in es:
            x={
                "note_id":nid,
                "text":e.get("text",""),
                "snomed_id":str(e.get(
                    "snomed_id",
                    e.get("snomed_ct_id",e.get("concept_id",""))
                ))
            }
            if "start" in e and "end" in e:
                x["start"]=int(e["start"])
                x["end"]=int(e["end"])
            out.append(x)
    return out

def prf(g,p):
    tp=len(g&p); fp=len(p-g); fn=len(g-p)
    P=tp/(tp+fp) if tp+fp else 0
    R=tp/(tp+fn) if tp+fn else 0
    F=2*P*R/(P+R) if P+R else 0
    return P,R,F

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--gold",default="../outputs/test_gold.jsonl")
    a.add_argument("--pred",required=True)
    a.add_argument("--format",choices=["standard","qwen"],default="standard")
    a=a.parse_args()

    G=gold(a.gold)
    test_ids={x["note_id"] for x in G}
    P=[x for x in pred(a.pred,a.format) if x["note_id"] in test_ids]

    if not all("start" in x and "end" in x for x in P):
        print("Predictions do not contain offsets; exact-span metrics unavailable.")
        return

    gs={(x["note_id"],x["start"],x["end"]) for x in G}
    ps={(x["note_id"],x["start"],x["end"]) for x in P}
    ep,er,ef=prf(gs,ps)

    gm={(x["note_id"],x["start"],x["end"]):x for x in G}
    pm={(x["note_id"],x["start"],x["end"]):x for x in P}
    common=gs&ps

    correct=sum(gm[k]["snomed_id"]==pm[k]["snomed_id"] for k in common)
    acc=correct/len(common) if common else 0

    ge={(x["note_id"],x["start"],x["end"],x["snomed_id"]) for x in G}
    pe={(x["note_id"],x["start"],x["end"],x["snomed_id"]) for x in P}
    xp,xr,xf=prf(ge,pe)

    print(f"Gold entities      : {len(G)}")
    print(f"Predicted entities : {len(P)}")
    print()
    print("Entity Recognition")
    print(f"Precision : {ep:.4f}")
    print(f"Recall    : {er:.4f}")
    print(f"F1        : {ef:.4f}")
    print()
    print("Concept Linking")
    print(f"Correct   : {correct}/{len(common)}")
    print(f"Accuracy  : {acc:.4f}")
    print()
    print("End-to-End")
    print(f"Precision : {xp:.4f}")
    print(f"Recall    : {xr:.4f}")
    print(f"F1        : {xf:.4f}")

if __name__=="__main__":
    main()
