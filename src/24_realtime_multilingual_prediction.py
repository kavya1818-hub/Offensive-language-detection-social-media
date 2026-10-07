# ==============================================================================
# STEP 24 - FINAL CROSS-DATASET CONTEXT-AWARE SEMANTIC PREDICTION
# CONTROLLED RESTORATION: original LR + validation-calibrated hybrid
# Only added: English contextual augmentation. Step 20 remains frozen.
# No keyword rules / dictionary / manual sentence rules.
# ===============================================================================
import os, sys, json, random, warnings
import numpy as np
import pandas as pd
import torch, joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
warnings.filterwarnings('ignore')

SEED=42
CPU_THREADS=min(4, os.cpu_count() or 4)
MAX_LENGTH=96
BATCH_SIZE=8
TARGET_PER_CLASS=1000
CONTEXT_AUGMENTATION_SIZE=1000
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(CPU_THREADS)
try: torch.set_num_interop_threads(max(1,min(2,CPU_THREADS)))
except RuntimeError: pass
DEVICE=torch.device('cpu')
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT=os.path.join(BASE,'results','multilingual_cross_dataset_final')
MODEL_DIR=os.path.join(ROOT,'final_model')
D1=os.path.join(BASE,'data','train.csv')
D2=os.path.join(BASE,'data','dravidian_clean','dravidian_multilingual_clean.csv')
OUT=os.path.join(ROOT,'step24_semantic_final')
EMB=os.path.join(OUT,'embedding_cache'); SEM=os.path.join(OUT,'semantic_model')
SEM_FILE=os.path.join(SEM,'semantic_classifier.joblib')
CAL_FILE=os.path.join(OUT,'hybrid_calibration.json')
CFG=os.path.join(OUT,'step24_config.json')
for p in (OUT,EMB,SEM): os.makedirs(p,exist_ok=True)
LANGS=['English','Tamil','Malayalam','Kannada']; LABELS={0:'Non_Offensive',1:'Offensive'}
TRAIN='--train' in sys.argv

print('\n'+'='*80); print('STEP 24 - CONTROLLED FINAL CROSS-DATASET PREDICTION'); print('='*80)
print('Mode:', 'ONE-TIME TRAIN' if TRAIN else 'REAL-TIME'); print('CPU threads:',CPU_THREADS); print('Step 20: FROZEN'); print('Rules: NONE\n')

def textnorm(x): return '' if pd.isna(x) else ' '.join(str(x).strip().split())
def splitnorm(x):
    if pd.isna(x): return 'unknown'
    x=str(x).strip().lower()
    return {'training':'train','tr':'train','train':'train','validation':'validation','valid':'validation','val':'validation','dev':'validation','test':'test','testing':'test','te':'test'}.get(x,x)
def langnorm(x):
    if pd.isna(x): return 'Unknown'
    x=str(x).strip().lower()
    if 'tamil' in x:return 'Tamil'
    if 'malayalam' in x:return 'Malayalam'
    if 'kannada' in x:return 'Kannada'
    if x in ('english','eng'):return 'English'
    return 'Unknown'
def d1label(x):
    if pd.isna(x):return np.nan
    x=str(x).strip()
    if x=='Non_Toxic':return 0
    return 1 if x in {'Homophobia_LGBTQphobia','Misogyny_Sexism','Racism_Xenophobia','Harassment_Stalking','Religious_Hate','Profanity_Insult'} else np.nan
def d2label(x):
    if pd.isna(x):return np.nan
    s=str(x).strip().lower()
    if s in {'0','0.0','non_offensive','non-offensive','not_offensive','not offensive'}:return 0
    if s in {'1','1.0','offensive'}:return 1
    try:
        n=int(float(x)); return n if n in (0,1) else np.nan
    except: return np.nan
def textset(df): return set(df.text.astype(str).str.lower().str.strip())

if not os.path.isdir(MODEL_DIR): raise FileNotFoundError(MODEL_DIR)
tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(MODEL_DIR,local_files_only=True).to(DEVICE); model.eval()
print('Frozen Step 20 XLM-R loaded.\n')

def xlmr_prob(texts,show=True):
    out=[]
    for s in range(0,len(texts),BATCH_SIZE):
        enc=tokenizer(texts[s:s+BATCH_SIZE],padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors='pt')
        with torch.no_grad(): p=torch.softmax(model(**enc).logits,dim=-1)[:,1]
        out.extend(p.numpy().tolist()); n=min(s+BATCH_SIZE,len(texts))
        if show and (n%200==0 or n==len(texts)): print(f'Processed {n}/{len(texts)}')
    return np.asarray(out,dtype=np.float32)

def embed(texts,cache=None,show=True):
    if cache and os.path.exists(cache):
        try:
            a=np.load(cache)
            if a.shape==(len(texts),1536): return a.astype(np.float32)
        except: pass
    all_e=[]
    for s in range(0,len(texts),BATCH_SIZE):
        enc=tokenizer(texts[s:s+BATCH_SIZE],padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors='pt')
        with torch.no_grad(): h=model.roberta(**enc).last_hidden_state
        mask=enc['attention_mask'].unsqueeze(-1).float(); mh=h*mask
        mean=mh.sum(1)/mask.sum(1).clamp(min=1e-9)
        mx=h.masked_fill(mask==0,torch.finfo(h.dtype).min).max(1).values
        all_e.append(torch.cat([mean,mx],1).numpy().astype(np.float32))
        n=min(s+BATCH_SIZE,len(texts))
        if show and (n%200==0 or n==len(texts)): print(f'Processed {n}/{len(texts)}')
    a=np.vstack(all_e).astype(np.float32)
    if cache: np.save(cache,a)
    return a

def load_data():
    a=pd.read_csv(D1); a['text']=a.text.apply(textnorm); a['label_id']=a.label.apply(d1label); a['language_clean']='English'
    a=a[(a.text.str.len()>0)&a.label_id.notna()].copy(); a.label_id=a.label_id.astype(int); a=a.drop_duplicates('text').reset_index(drop=True)
    b=pd.read_csv(D2); req={'text','mapped_label','language','split'}
    if not req.issubset(b.columns): raise ValueError(f'Dataset 2 missing {req-set(b.columns)}')
    b['text']=b.text.apply(textnorm); b['label_id']=b.mapped_label.apply(d2label); b['language_clean']=b.language.apply(langnorm); b['split_clean']=b.split.apply(splitnorm)
    b=b[(b.text.str.len()>0)&b.label_id.notna()&b.language_clean.isin(LANGS[1:])].copy(); b.label_id=b.label_id.astype(int); b=b.drop_duplicates('text').reset_index(drop=True)
    a_tr,a_tmp=train_test_split(a,test_size=.2,stratify=a.label_id,random_state=SEED); a_val,a_te=train_test_split(a_tmp,test_size=.5,stratify=a_tmp.label_id,random_state=SEED)
    b_tr=b[b.split_clean=='train'].copy(); b_val=b[b.split_clean=='validation'].copy(); b_te=b[b.split_clean=='test'].copy()
    ts=textset(b_tr); b_val=b_val[~b_val.text.str.lower().str.strip().isin(ts)].copy(); vs=textset(b_val); b_te=b_te[~b_te.text.str.lower().str.strip().isin(ts)&~b_te.text.str.lower().str.strip().isin(vs)].copy()
    assert not(textset(b_tr)&textset(b_val)); assert not(textset(b_tr)&textset(b_te)); assert not(textset(b_val)&textset(b_te))
    print('Dataset 1:',len(a_tr),len(a_val),len(a_te)); print('Dataset 2:',len(b_tr),len(b_val),len(b_te)); print('DATA LEAKAGE CHECK PASSED.')
    return a_tr,a_val,a_te,b_tr,b_val,b_te

def sample(df,language,label,n,seed):
    x=df[(df.language_clean==language)&(df.label_id==label)]
    if len(x)==0: raise ValueError(f'No {language} label {label} examples')
    return x.sample(n=n,replace=len(x)<n,random_state=seed)

def semantic_set(a,b):
    parts=[]
    for i,(l,df) in enumerate([('English',a),('Tamil',b),('Malayalam',b),('Kannada',b)]):
        for y in (0,1): parts.append(sample(df,l,y,TARGET_PER_CLASS,SEED+i*10+y))
    return pd.concat(parts,ignore_index=True).drop_duplicates('text').sample(frac=1,random_state=SEED).reset_index(drop=True)

def augment(a):
    src=a[a.label_id==1].sample(n=min(CONTEXT_AUGMENTATION_SIZE,len(a[a.label_id==1])),random_state=SEED+500)
    templates=[
      'The teacher explained why the following sentence is offensive: "{text}"',
      'The article discusses the following sentence as an example of offensive language: "{text}"',
      'The following sentence is quoted in a lesson about offensive language: "{text}"',
      'A researcher analyzed the following sentence to study offensive language: "{text}"',
      'The following sentence is being discussed as an example, not used to insult anyone: "{text}"',
      'The speaker was discussing the meaning and use of this sentence: "{text}"',
      'This sentence appears as a quoted example in a discussion of language: "{text}"',
      'The following text is being analyzed for its language use: "{text}"']
    return pd.DataFrame([{'text':templates[i%len(templates)].format(text=r.text),'label_id':0,'language_clean':'English'} for i,r in enumerate(src.itertuples())])

def metrics(name,y,p):
    pr,re,f,_=precision_recall_fscore_support(y,p,labels=[1],average=None,zero_division=0)
    print(f'\n{name}\nAccuracy: {accuracy_score(y,p):.4f}\nMacro-F1: {f1_score(y,p,average="macro"):.4f}\nOffensive Precision: {pr[0]:.4f}\nOffensive Recall: {re[0]:.4f}\nOffensive F1: {f[0]:.4f}')

def train():
    a_tr,a_val,a_te,b_tr,b_val,b_te=load_data()
    sem=semantic_set(a_tr,b_tr)
    aug=augment(a_tr)
    sem=pd.concat([sem,aug],ignore_index=True).drop_duplicates('text').sample(frac=1,random_state=SEED).reset_index(drop=True)
    print('\nSemantic training:',len(sem)); print(sem.groupby(['language_clean','label_id']).size())
    X=embed(sem.text.tolist(),os.path.join(EMB,'semantic_train_final.npy')); y=sem.label_id.to_numpy(dtype=np.int64)
    clf=Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression(C=.1,max_iter=1000,class_weight='balanced',random_state=SEED))])
    clf.fit(X,y); joblib.dump(clf,SEM_FILE); print('Semantic model saved:',SEM_FILE)
    v=pd.concat([a_val[['text','label_id','language_clean']],b_val[['text','label_id','language_clean']]],ignore_index=True).drop_duplicates('text').reset_index(drop=True)
    vt=v.text.astype(str).str.lower().str.strip(); st=textset(sem); v=v[~vt.isin(st)].reset_index(drop=True)
    VX=embed(v.text.tolist(),os.path.join(EMB,'official_validation_final.npy')); vs=clf.predict_proba(VX)[:,1]; vx=xlmr_prob(v.text.tolist())
    yv=v.label_id.to_numpy(dtype=np.int64)
    best=None
    for th in np.arange(.30,.71,.01):
        for w in np.arange(.20,.81,.05):
            hp=w*vx+(1-w)*vs; pred=(hp>=th).astype(int)
            mf=f1_score(yv,pred,average='macro'); acc=accuracy_score(yv,pred)
            pr,re,ff,_=precision_recall_fscore_support(yv,pred,labels=[1],average=None,zero_division=0)
            c={'threshold':float(th),'xlmr_weight':float(w),'semantic_weight':float(1-w),'accuracy':float(acc),'macro_f1':float(mf),'offensive_precision':float(pr[0]),'offensive_recall':float(re[0]),'offensive_f1':float(ff[0])}
            if best is None or c['macro_f1']>best['macro_f1']: best=c
    with open(CAL_FILE,'w',encoding='utf8') as f: json.dump(best,f,indent=2)
    print('\nBest threshold:',f"{best['threshold']:.2f}"); print('XLM-R weight:',f"{best['xlmr_weight']:.2f}"); print('Semantic weight:',f"{best['semantic_weight']:.2f}")
    hp=best['xlmr_weight']*vx+best['semantic_weight']*vs; pred=(hp>=best['threshold']).astype(int); metrics('OFFICIAL VALIDATION',yv,pred)
    with open(CFG,'w',encoding='utf8') as f: json.dump({'step':24,'version':'controlled_final','step20_frozen':True,'languages':LANGS,'representation':'mean_max_xlmr_contextual_embedding','classifier':'StandardScaler_LogisticRegression','contextual_english_augmentation':True,'keyword_rules':False,'offensive_dictionary':False,'manual_sentence_rules':False,'test_used_for_training':False,'threshold':best['threshold'],'xlmr_weight':best['xlmr_weight'],'semantic_weight':best['semantic_weight']},f,indent=2)
    print('\n'+'='*80); print('STEP 24 CONTROLLED FINAL TRAINING COMPLETE'); print('='*80); print('Run WITHOUT --train now.')

def detect_language(s):
    for c in s:
        n=ord(c)
        if 0x0C80<=n<=0x0CFF:return 'Kannada'
        if 0x0D00<=n<=0x0D7F:return 'Malayalam'
        if 0x0B80<=n<=0x0BFF:return 'Tamil'
    return 'English'

def realtime():
    if not(os.path.exists(SEM_FILE) and os.path.exists(CAL_FILE) and os.path.exists(CFG)): raise FileNotFoundError('Final Step 24 artifacts not found. Run with --train once.')
    clf=joblib.load(SEM_FILE)
    with open(CAL_FILE,encoding='utf8') as f: cal=json.load(f)
    print('\nREAL-TIME MODE: XLM-R + CONTEXTUAL SEMANTIC CLASSIFIER')
    print('No keyword rules | No dictionary | No manual sentence rules'); print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:s=input('Enter text: ')
        except(EOFError,KeyboardInterrupt): print(); break
        if s.strip().lower() in ('exit','quit'): break
        s=textnorm(s)
        if not s: continue
        lang=detect_language(s); xp=float(xlmr_prob([s],False)[0]); sp=float(clf.predict_proba(embed([s],None,False))[0,1]); hp=float(cal['xlmr_weight']*xp+cal['semantic_weight']*sp); th=float(cal['threshold']); pred=1 if hp>=th else 0; conf=hp if pred else 1-hp
        print('\n'+'-'*64); print('PREDICTION RESULT'); print('-'*64); print('Language       :',lang); print('Prediction     :',LABELS[pred]); print(f'Confidence     : {conf*100:.2f}%'); print(f'Non_Offensive  : {(1-hp)*100:.2f}%'); print(f'Offensive      : {hp*100:.2f}%'); print(f'XLM-R score    : {xp*100:.2f}%'); print(f'Semantic score : {sp*100:.2f}%'); print(f'Hybrid score   : {hp*100:.2f}%'); print('-'*64)

if TRAIN: train()
else: realtime()
