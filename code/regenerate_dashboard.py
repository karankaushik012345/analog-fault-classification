import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

REPO = r"C:\Users\karan\Desktop\fault-classification-project"
OUT_DIR = os.path.join(REPO, "results")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [f'f{i}' for i in range(1,18)]
FEATURE_NAMES = ['peak_voltage','rms_value','mean_voltage','std_voltage',
    'rise_time','settling_time','skewness','kurtosis','dominant_freq',
    'spectral_energy','spec_centroid','bandwidth','spec_entropy','energy',
    'zero_crossing_rate','crest_factor','dc_level']

df = pd.read_csv(os.path.join(REPO, "data", "features_v2.csv"))
df['label'] = df['circuit'] + '_' + df['fault']
le = LabelEncoder()
y = le.fit_transform(df['label'])
X = df[FEATURE_COLS].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)
labels = list(le.classes_)

rc_open_idx  = labels.index('RC_open')
rlc_open_idx = labels.index('RLC_open')
test_indices = list(pd.Series(range(len(df))).sample(frac=0.2, random_state=42).sort_values().index)
test_circuits = df.iloc[test_indices]['circuit'].values

def apply_topology_rule(pred_array):
    pred = pred_array.tolist()
    for i in range(len(pred)):
        if pred[i] == rlc_open_idx and test_circuits[i] == 'RC':
            pred[i] = rc_open_idx
    return np.array(pred)

print('Training models...')
configs = {
    'Linear\n(default)':  SVC(kernel='linear', C=1.0, decision_function_shape='ovo', random_state=42),
    'RBF\n(default)':     SVC(kernel='rbf',    C=1.0, gamma='scale', decision_function_shape='ovo', random_state=42),
    'Poly\n(default)':    SVC(kernel='poly',   C=1.0, gamma='scale', degree=3, coef0=1, decision_function_shape='ovo', random_state=42),
    'Linear\n(tuned)':    SVC(kernel='linear', C=100, decision_function_shape='ovo', random_state=42),
    'RBF\n(tuned)':       SVC(kernel='rbf',    C=100, gamma=0.1, decision_function_shape='ovo', random_state=42),
    'Poly\n(tuned)':      SVC(kernel='poly',   C=100, gamma=0.1, degree=3, coef0=1, decision_function_shape='ovo', random_state=42),
}
accs = {}
for name, clf in configs.items():
    clf.fit(X_tr, y_tr)
    pred = apply_topology_rule(clf.predict(X_te))
    accs[name] = accuracy_score(y_te, pred)*100
    print(f'  {name.replace(chr(10)," "):<22}: {accs[name]:.2f}%')

best_clf  = configs['RBF\n(tuned)']
best_pred = apply_topology_rule(best_clf.predict(X_te))
best_acc  = accs['RBF\n(tuned)']
cm        = confusion_matrix(y_te, best_pred)

lin_clf    = configs['Linear\n(tuned)']
importance = np.mean(np.abs(lin_clf.coef_), axis=0)
feat_imp_df = pd.DataFrame({'feature':FEATURE_NAMES,'importance':importance}).sort_values('importance',ascending=True)

y_test_labels = le.inverse_transform(y_te)
y_pred_labels = le.inverse_transform(best_pred)
circuit_accs  = {}
for circ in ['RC','RLC','SALLEN_KEY','RC_LADDER']:
    mask = np.array([circ+'_' in l for l in y_test_labels])
    if mask.sum()>0:
        circuit_accs[circ] = np.sum(y_test_labels[mask]==y_pred_labels[mask])/mask.sum()*100

print('\nBuilding final dashboard...')
fig = plt.figure(figsize=(24,20))
fig.patch.set_facecolor('#f8f9fa')
gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.35)
TK  = dict(fontsize=13,fontweight='bold',pad=10)
C6  = ['#90caf9','#90caf9','#90caf9','#1565c0','#1565c0','#1565c0']

ax1 = fig.add_subplot(gs[0,0])
names=list(accs.keys()); vals=list(accs.values())
bars=ax1.bar(names,vals,color=C6,edgecolor='white',linewidth=0.8,width=0.6)
ax1.axhline(y=best_acc,color='#d32f2f',linestyle='--',linewidth=1.5,label=f'Best {best_acc:.1f}%')
ax1.set_ylim(60,95); ax1.set_ylabel('Test Accuracy (%)',fontsize=10)
ax1.set_title('Kernel Comparison\nDefault vs Tuned + Topology Rule',**TK)
ax1.legend(fontsize=8)
for bar,val in zip(bars,vals):
    ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f'{val:.1f}%',ha='center',va='bottom',fontsize=8,fontweight='bold')
ax1.tick_params(axis='x',labelsize=8); ax1.set_facecolor('#ffffff')

ax2 = fig.add_subplot(gs[0,1])
cn=list(circuit_accs.keys()); cv=list(circuit_accs.values())
cc=['#ef9a9a' if v<70 else '#fff176' if v<85 else '#a5d6a7' for v in cv]
bars2=ax2.bar(cn,cv,color=cc,edgecolor='grey',linewidth=0.8,width=0.5)
ax2.set_ylim(0,115); ax2.set_ylabel('Accuracy (%)',fontsize=10)
ax2.set_title('Per-Circuit Accuracy\n(RBF Tuned + Topology Rule)',**TK)
for bar,val in zip(bars2,cv):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{val:.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
ax2.tick_params(axis='x',labelsize=9); ax2.set_facecolor('#ffffff')

ax3 = fig.add_subplot(gs[0,2])
fc=df.groupby('fault').size()
ax3.pie(fc.values,labels=fc.index,autopct='%1.1f%%',
        colors=['#ef5350','#42a5f5','#66bb6a','#ffa726','#ab47bc'],
        startangle=90,textprops={'fontsize':9})
ax3.set_title(f'Dataset Composition\n({len(df)} total samples)',**TK)

ax4 = fig.add_subplot(gs[1,0])
bc=['#1565c0' if v>=feat_imp_df['importance'].quantile(0.75) else '#90caf9' for v in feat_imp_df['importance']]
ax4.barh(feat_imp_df['feature'],feat_imp_df['importance'],color=bc,edgecolor='white')
ax4.set_xlabel('Mean |Coefficient|',fontsize=10)
ax4.set_title('Feature Importance\n(Linear SVM Weights)',**TK)
ax4.tick_params(axis='y',labelsize=8); ax4.set_facecolor('#ffffff')

ax5 = fig.add_subplot(gs[1,1:])
f1s=f1_score(y_te,best_pred,average=None)
f1_df=pd.DataFrame({'class':labels,'f1':f1s}).sort_values('f1',ascending=True)
f1c=['#d32f2f' if f==0 else '#ef9a9a' if f<0.5 else '#fff176' if f<0.8 else '#a5d6a7' for f in f1_df['f1']]
ax5.barh(f1_df['class'],f1_df['f1'],color=f1c,edgecolor='white',height=0.7)
ax5.axvline(x=0.5,color='orange',linestyle='--',alpha=0.7,linewidth=1.5)
ax5.axvline(x=0.8,color='green',linestyle='--',alpha=0.7,linewidth=1.5)
for i,(_,row) in enumerate(f1_df.iterrows()):
    ax5.text(row['f1']+0.01,i,str(round(row['f1'],2)),va='center',fontsize=8)
ax5.set_xlabel('F1 Score',fontsize=10); ax5.set_xlim(0,1.2)
ax5.set_title('Per-Class F1 — RBF + Topology Rule (Final)',**TK)
ax5.set_facecolor('#ffffff')

ax6 = fig.add_subplot(gs[2,:])
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,
            linewidths=0.3,ax=ax6,cbar_kws={'shrink':0.8},annot_kws={'size':7})
ax6.set_xlabel('Predicted',fontsize=11); ax6.set_ylabel('Actual',fontsize=11)
ax6.set_title('Confusion Matrix — RBF + Topology Rule | Accuracy: ' + str(round(best_acc,2)) + '%',**TK)
ax6.tick_params(axis='x',labelsize=7,rotation=45)
ax6.tick_params(axis='y',labelsize=7,rotation=0)

# CORRECTED TITLE - SN Bose Summer Internship, NIT Silchar (not Research Centre)
fig.suptitle('Kernel-Based Fault Classification in Analog Circuits\nSN Bose Summer Internship 2026, NIT Silchar — FINAL Results (78.43%, Topology-Aware)',
             fontsize=16,fontweight='bold',y=0.98)

out_path = os.path.join(OUT_DIR,'results_dashboard_FINAL.png')
plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor=fig.get_facecolor())
plt.close()
print('Dashboard saved -> ' + out_path)
